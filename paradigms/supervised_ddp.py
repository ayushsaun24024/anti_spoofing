import os
import torch
import datetime
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from architecture.nasnet import nasnet_audio
from utilities.engine import read_configs, Logs
from dataset.asvspoof import ASVspoofDataset, SASVDataset_v1
from metrics.confusion_matrix import AntiSpoofing_CM_Metrics, AntiSpoofing_SASV_Metrics


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=7200))


def main(rank, world_size):
    ddp_setup(rank, world_size)
    
    config, exp_name = read_configs()
    logger = Logs(basepath="/home/abrol/anti_spoofing", exp_name=exp_name) if rank == 0 else None
    DEVICE = torch.device(f"cuda:{rank}")
    
    train_dataset = ASVspoofDataset(DATA_PATH_LIST=config["TRAIN_AUDIO_PATH"],
                                   PROTOCOL_PATH_LIST=config["TRAIN_PROTOCOL_PATH"],
                                   split="train",
                                   noise_type=config["NOISE_TYPE"])
    
    test_dataset = ASVspoofDataset(DATA_PATH_LIST=config["EVAL_AUDIO_PATH"],
                                  PROTOCOL_PATH_LIST=config["EVAL_PROTOCOL_PATH"],
                                  split="test",
                                  noise_type=None)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_dataloader = DataLoader(dataset=train_dataset,
                                 batch_size=config["BATCH_SIZE"],
                                 sampler=train_sampler,  
                                 num_workers=2,
                                 pin_memory=True)
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=config["BATCH_SIZE"],
                                sampler=test_sampler,  
                                num_workers=2,
                                pin_memory=True)

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

    architecture = nasnet_audio(num_classes=2)
    architecture = architecture.to(DEVICE)
    
    architecture = DDP(architecture, device_ids=[rank])

    weight = torch.FloatTensor([0.1, 0.9]).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(architecture.parameters(),
                                lr=config["LR"],
                                betas=[0.9, 0.999],
                                weight_decay=config["WEIGHT_DECAY"],
                                amsgrad=False)

    total_steps = config['EPOCHS'] * len(train_dataloader)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(step, total_steps, 1, 0.05))

    best_loss = [99]
    for epoch in range(config["EPOCHS"]):
        train_sampler.set_epoch(epoch)
        
        architecture.train()
        
        epoch_loss, epoch_accuracy = 0, 0
        tpredlogit = []
        tpred = []
        tlabel_list = []
        
        for tmb_idx, (tdata, tlabel, _) in enumerate(train_dataloader):
            batch_size = tlabel.size(0)
            
            optimizer.zero_grad()
            tpredictions = architecture(tdata.to(DEVICE))
            tspf_loss = loss_fn(tpredictions, tlabel.to(DEVICE))
            tloss = tspf_loss
            tloss.backward()
            optimizer.step()
            scheduler.step()
            
            taccuracy = (F.softmax(tpredictions, dim=1).argmax(dim=1).detach().cpu()==tlabel).sum().item() / batch_size * 100
            
            if rank == 0 and tmb_idx % config["MINI_BATCH_SIZE"] == 0:
                logger.write(
                    "Epoch: [{}], MiniBatch: [{}], Loss: [{}], Spoof/Bonafide Accuracy: [{}]".format(
                        epoch+1, tmb_idx+1, tloss.item(), taccuracy
                    )
                )
            
            epoch_loss += tloss.item()
            epoch_accuracy += taccuracy
            
            tpredlogit.extend(tpredictions.detach().cpu().numpy())
            tpred.extend(F.softmax(tpredictions, dim=1).detach().cpu().numpy())
            tlabel_list.extend(tlabel.numpy())
        
        all_tpredlogit = [None for _ in range(world_size)]
        all_tpred = [None for _ in range(world_size)]
        all_tlabel_list = [None for _ in range(world_size)]
        
        torch.distributed.all_gather_object(all_tpredlogit, tpredlogit)
        torch.distributed.all_gather_object(all_tpred, tpred)
        torch.distributed.all_gather_object(all_tlabel_list, tlabel_list)
        
        combined_tpredlogit = [item for sublist in all_tpredlogit for item in sublist]
        combined_tpred = [item for sublist in all_tpred for item in sublist]
        combined_tlabel_list = [item for sublist in all_tlabel_list for item in sublist]
        
        avg_loss = torch.tensor([epoch_loss]).to(DEVICE)
        avg_accuracy = torch.tensor([epoch_accuracy]).to(DEVICE)
        
        torch.distributed.all_reduce(avg_loss)
        torch.distributed.all_reduce(avg_accuracy)
        
        avg_loss = avg_loss.item() / world_size / len(train_dataloader)
        avg_accuracy = avg_accuracy.item() / world_size / len(train_dataloader)
        
        tmetric = AntiSpoofing_CM_Metrics(
            predicted_logits=np.array(combined_tpredlogit), 
            prediction=np.array(combined_tpred), 
            label=np.array(combined_tlabel_list)
        )
        
        if rank == 0:
        
            logger.write(
                " >> >> >> [Training] Epoch: [{}], Loss: [{}], Spoof/Bonafide Accuracy: [{}]".format(
                    epoch + 1, avg_loss, avg_accuracy
                )
            )
            
            logger.write(
                " >> >> >> [Training] Epoch: [{}], minDCF: [{}], actDCF: [{}], EER: [{}]".format(
                    epoch + 1,
                    tmetric.compute_mindcf(0.05, 1, 10),
                    tmetric.compute_actDCF(0.05, 1, 10),
                    tmetric.compute_EER(),
                )
            )
            
            if avg_loss < best_loss[-1]:
                torch.save(
                    {
                        "architecture": architecture.module.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(logger.chkpt_folder, "best_model.pth")
                )
                best_loss.append(avg_loss)
        
        torch.distributed.barrier()
        
        if (epoch + 1) % config["EVALUATE_AFTER"] == 0:
            architecture.eval()
            torch.cuda.empty_cache()
            
            epoch_eloss, epoch_eaccuracy = 0, 0
            epredlogit = []
            epred = []
            elabel_list = []
            
            with torch.no_grad():
                for emb_idx, (edata, elabel, _) in enumerate(test_dataloader):
                    batch_size = elabel.size(0)
                    epredictions = architecture(edata.to(DEVICE))
                    espf_loss = loss_fn(epredictions, elabel.to(DEVICE))
                    eaccuracy = (F.softmax(epredictions, dim=1).argmax(dim=1).detach().cpu()==elabel).sum().item() / batch_size * 100
                    
                    epoch_eloss += espf_loss.item()
                    epoch_eaccuracy += eaccuracy
                    
                    epredlogit.extend(epredictions.detach().cpu().numpy())
                    epred.extend(F.softmax(epredictions, dim=1).detach().cpu().numpy())
                    elabel_list.extend(elabel.numpy())
                
                all_epredlogit = [None for _ in range(world_size)]
                all_epred = [None for _ in range(world_size)]
                all_elabel_list = [None for _ in range(world_size)]
                
                torch.distributed.all_gather_object(all_epredlogit, epredlogit)
                torch.distributed.all_gather_object(all_epred, epred)
                torch.distributed.all_gather_object(all_elabel_list, elabel_list)
                
                combined_epredlogit = [item for sublist in all_epredlogit for item in sublist]
                combined_epred = [item for sublist in all_epred for item in sublist]
                combined_elabel_list = [item for sublist in all_elabel_list for item in sublist]
                
                avg_eloss = torch.tensor([epoch_eloss]).to(DEVICE)
                avg_eaccuracy = torch.tensor([epoch_eaccuracy]).to(DEVICE)
                
                torch.distributed.all_reduce(avg_eloss)
                torch.distributed.all_reduce(avg_eaccuracy)
                
                avg_eloss = avg_eloss.item() / world_size / len(test_dataloader)
                avg_eaccuracy = avg_eaccuracy.item() / world_size / len(test_dataloader)
                
                emetric = AntiSpoofing_CM_Metrics(
                    predicted_logits=np.array(combined_epredlogit), 
                    prediction=np.array(combined_epred), 
                    label=np.array(combined_elabel_list)
                )
                
                if rank == 0:
                    logger.write(
                        " >> >> >> [Testing CM] Epoch: [{}], Loss: [{}], Spoof/Bonafide Accuracy: [{}]".format(
                            epoch + 1, avg_eloss, avg_eaccuracy
                        )
                    )
                    
                    logger.write(
                        " >> >> >> [Testing CM] Epoch: [{}], minDCF: [{}], actDCF: [{}], EER: [{}]".format(
                            epoch + 1,
                            emetric.compute_mindcf(0.05, 1, 10),
                            emetric.compute_actDCF(0.05, 1, 10),
                            emetric.compute_EER(),
                        )
                    )
            
            torch.distributed.barrier()
    
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

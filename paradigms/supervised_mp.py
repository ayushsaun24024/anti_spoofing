import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast  # Added for mixed precision

from dataset.asvspoof import ASVspoofDataset
from utilities.wandb_logger import WandBLogger
from utilities.engine import read_configs, Logs
from metrics.confusion_matrix import AntiSpoofing_CM_Metrics
from architecture.wavlm_nes2net_x import WavLM_Nes2Net_noRes_w_allT

config, exp_name = read_configs()
logger = Logs(basepath="/home/abrol/anti_spoofing", exp_name=exp_name)
DEVICE = torch.device("cuda:{}".format(config["CUDA_DEVICE"]))

wandb_logger = WandBLogger(
    project_name=exp_name,
    experiment_name=exp_name,
    config={}
)

train_dataset = ASVspoofDataset(DATA_PATH_LIST=config["TRAIN_AUDIO_PATH"],
                                PROTOCOL_PATH_LIST=config["TRAIN_PROTOCOL_PATH"],
                                noise_dict={
                                    "musan_path": config["MUSAN_PATH"],
                                    "rir_path": config["RIR_PATH"]
                                },
                                split="train",
                                noise_type=config["NOISE_TYPE"])
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=config["BATCH_SIZE"],
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
validation_dataset = ASVspoofDataset(DATA_PATH_LIST=config["DEV_AUDIO_PATH"],
                               PROTOCOL_PATH_LIST=config["DEV_PROTOCOL_PATH"],
                                noise_dict={
                                    "musan_path": None,
                                    "rir_path": None
                                },
                               split="test",
                               noise_type=None)
validation_dataloader = DataLoader(dataset=validation_dataset,
                             batch_size=config["BATCH_SIZE"],
                             num_workers=2,
                             pin_memory=True)
test_dataset = ASVspoofDataset(DATA_PATH_LIST=config["EVAL_AUDIO_PATH"],
                               PROTOCOL_PATH_LIST=config["EVAL_PROTOCOL_PATH"],
                                noise_dict={
                                    "musan_path": None,
                                    "rir_path": None
                                },
                               split="test",
                               noise_type=None)
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=config["BATCH_SIZE"],
                             num_workers=2,
                             pin_memory=True)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

architecture = WavLM_Nes2Net_noRes_w_allT(DEVICE)

weight = torch.tensor([0.1, 0.9], dtype=torch.float32).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

architecture.to(DEVICE)

wandb_logger.log_model_architecture(architecture)

optimizer = torch.optim.Adam(architecture.parameters(),
                             lr=config["LR"],
                             betas=[0.9, 0.999],
                             weight_decay=config["WEIGHT_DECAY"],
                             amsgrad=False)

# Initialize mixed precision scaler
scaler = GradScaler()

total_steps = config['EPOCHS'] * len(train_dataloader)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(step, total_steps, lr_max=1.0, lr_min=0.1))

best_loss = [99]
for epoch in range(config["EPOCHS"]):
    architecture.train()
    
    epoch_loss, epoch_accuracy = 0,0
    tpredlogit = []
    tpred = []
    tlabel_list = []
    for tmb_idx, (tdata, tlabel, tattack_label) in enumerate(train_dataloader):

        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            tpredictions = architecture(tdata.to(DEVICE))
            tspf_loss = loss_fn(tpredictions, tlabel.to(DEVICE))
            tloss = tspf_loss
        
        # Mixed precision backward pass
        scaler.scale(tloss).backward()
        
        wandb_logger.log_gradients(architecture, log_freq=100)
        
        # Mixed precision optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        wandb_logger.log_weights(architecture, log_freq=500)
        
        taccuracy = (F.softmax(tpredictions, dim=1).argmax(dim=1).detach().cpu()==tlabel).sum().div(config["BATCH_SIZE"]).mul(100)
        
        current_lr = scheduler.get_last_lr()[0]
        wandb_logger.log_training_batch(
            loss=tloss.item(),
            accuracy=taccuracy.item(),
            learning_rate=current_lr,
            predictions=tpredictions,
            labels=tlabel
        )
        
        wandb_logger.log_system_metrics()
        
        if tmb_idx%config["MINI_BATCH_SIZE"] == 0:
            logger.write(
                "Epoch: [{}], MiniBatch: [{}], Loss: [{}], Spoof/Bonafide Accuracy: [{}]".format(
                    epoch+1, tmb_idx+1, tloss.item(), taccuracy.item()
                )
            )
        epoch_loss += tloss.item()
        epoch_accuracy += taccuracy.item()
        tpredlogit.extend(tpredictions.detach().cpu().numpy())
        tpred.extend(F.softmax(tpredictions, dim=1).detach().cpu().numpy())
        tlabel_list.extend(tlabel.numpy())
        
    logger.write(
        " >> >> >> [Training] Epoch: [{}], Loss: [{}], Spoof/Bonafide Accuracy: [{}]".format(
            epoch + 1, epoch_loss/len(train_dataloader), epoch_accuracy/len(train_dataloader)
        )
    )
    tmetric = AntiSpoofing_CM_Metrics(predicted_logits=np.array(tpredlogit), prediction=np.array(tpred), label=np.array(tlabel_list))
    
    train_metrics = {
        "minDCF": tmetric.compute_mindcf(0.05, 1, 10),
        "actDCF": tmetric.compute_actDCF(0.05, 1, 10),
        "EER": tmetric.compute_EER()
    }
    
    logger.write(
        " >> >> >> [Training] Epoch: [{}], minDCF: [{}], actDCF: [{}], EER: [{}]".format(
            epoch + 1,
            train_metrics["minDCF"],
            train_metrics["actDCF"],
            train_metrics["EER"],
        )
    )
    
    wandb_logger.log_training_epoch(
        epoch=epoch + 1,
        avg_loss=epoch_loss/len(train_dataloader),
        avg_accuracy=epoch_accuracy/len(train_dataloader),
        metrics_dict=train_metrics
    )
    
    wandb_logger.log_confusion_matrix(
        predictions=np.array(tpred),
        labels=np.array(tlabel_list),
        class_names=["Bonafide", "Spoof"],
        prefix="train"
    )
    
    if epoch_loss/len(train_dataloader) < best_loss[-1]:
        torch.save(
            {
                "architecture": architecture.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),  # Save scaler state
            },
            os.path.join(logger.chkpt_folder, "best_model.pth")
        )
        
        best_loss.append(epoch_loss/len(train_dataloader))
        
    ################################################################################# VALIDATION CODE ##################################################################
    
    architecture.eval()
    epoch_vloss, epoch_vaccuracy = 0, 0
    
    vpredlogit = []
    vpred = []
    vlabel_list = []
    
    with torch.no_grad():
        for vmb_idx, (vdata, vlabel, vattack_label) in enumerate(validation_dataloader):
            # Mixed precision inference
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                vpredictions = architecture(vdata.to(DEVICE))
                vspf_loss = loss_fn(vpredictions, vlabel.to(DEVICE))
                
            vaccuracy = (F.softmax(vpredictions, dim=1).argmax(dim=1).detach().cpu()==vlabel).sum().div(config["BATCH_SIZE"]).mul(100)
            
            wandb_logger.log_validation_batch(
                loss=vspf_loss.item(),
                accuracy=vaccuracy.item(),
                predictions=vpredictions,
                labels=vlabel
            )
            
            epoch_vloss += vspf_loss.item()
            epoch_vaccuracy += vaccuracy.item()
            
            vpredlogit.extend(vpredictions.detach().cpu().numpy())
            vpred.extend(F.softmax(vpredictions, dim=1).detach().cpu().numpy())
            vlabel_list.extend(vlabel.numpy())
            
    logger.write(
        " >> >> >> [Validation] Epoch: [{}], Loss: [{}], Spoof/Bonafide Accuracy: [{}]".format(
            epoch + 1, epoch_vloss/len(validation_dataloader), epoch_vaccuracy/len(validation_dataloader)
        )
    )
    
    vmetric = AntiSpoofing_CM_Metrics(predicted_logits=np.array(vpredlogit), prediction=np.array(vpred), label=np.array(vlabel_list))
    
    validation_metrics = {
        "minDCF": vmetric.compute_mindcf(0.05, 1, 10),
        "actDCF": vmetric.compute_actDCF(0.05, 1, 10),
        "EER": vmetric.compute_EER()
    }
    
    logger.write(
        " >> >> >> [Validation] Epoch: [{}], minDCF: [{}], actDCF: [{}], EER: [{}]".format(
            epoch + 1,
            validation_metrics["minDCF"],
            validation_metrics["actDCF"],
            validation_metrics["EER"],
        )
    )
    
    wandb_logger.log_validation_epoch(
        epoch=epoch + 1,
        avg_loss=epoch_vloss/len(validation_dataloader),
        avg_accuracy=epoch_vaccuracy/len(validation_dataloader),
        metrics_dict=validation_metrics
    )
    
    wandb_logger.log_confusion_matrix(
        predictions=np.array(vpred),
        labels=np.array(vlabel_list),
        class_names=["Bonafide", "Spoof"],
        prefix="validation"
    )
    
    ################################################################################# EVALUATION CODE ##################################################################
        
    if (epoch + 1)%config["EVALUATE_AFTER"] == 0:
        architecture.eval()
        epoch_eloss, epoch_eaccuracy = 0,0
        
        epredlogit = []
        epred = []
        elabel_list = []
        
        with torch.no_grad():
            for emb_idx, (edata, elabel, eattack_label) in enumerate(test_dataloader):
                # Mixed precision inference
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    epredictions = architecture(edata.to(DEVICE))
                    espf_loss = loss_fn(epredictions, elabel.to(DEVICE))
                    
                eaccuracy = (F.softmax(epredictions, dim=1).argmax(dim=1).detach().cpu()==elabel).sum().div(config["BATCH_SIZE"]).mul(100)
                
                wandb_logger.log_testing_batch(
                    loss=espf_loss.item(),
                    accuracy=eaccuracy.item(),
                    predictions=epredictions,
                    labels=elabel
                )
                
                epoch_eloss += espf_loss.item()
                epoch_eaccuracy += eaccuracy.item()
                
                epredlogit.extend(epredictions.detach().cpu().numpy())
                epred.extend(F.softmax(epredictions, dim=1).detach().cpu().numpy())
                elabel_list.extend(elabel.numpy())
                
            logger.write(
                " >> >> >> [Testing CM] Epoch: [{}], Loss: [{}], Spoof/Bonafide Accuracy: [{}]".format(
                epoch + 1, epoch_eloss/len(test_dataloader), epoch_eaccuracy/len(test_dataloader)
                )
            )
            emetric = AntiSpoofing_CM_Metrics(predicted_logits=np.array(epredlogit), prediction=np.array(epred), label=np.array(elabel_list))
            
            eval_metrics = {
                "minDCF": emetric.compute_mindcf(0.05, 1, 10),
                "actDCF": emetric.compute_actDCF(0.05, 1, 10),
                "EER": emetric.compute_EER()
            }
            
            logger.write(
                " >> >> >> [Testing CM] Epoch: [{}], minDCF: [{}], actDCF: [{}], EER: [{}]".format(
                    epoch + 1,
                    eval_metrics["minDCF"],
                    eval_metrics["actDCF"],
                    eval_metrics["EER"],
                )
            )
            
            wandb_logger.log_evaluation(
                epoch=epoch + 1,
                avg_loss=epoch_eloss/len(test_dataloader),
                avg_accuracy=epoch_eaccuracy/len(test_dataloader),
                metrics_dict=eval_metrics
            )
            
            wandb_logger.log_confusion_matrix(
                predictions=np.array(epred),
                labels=np.array(elabel_list),
                class_names=["Bonafide", "Spoof"],
                prefix="eval"
            )

wandb_logger.finish()

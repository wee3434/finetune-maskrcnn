import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from get_data import CustData

class TrainModel(object):
    def __init__(self):
        self.annotation_dir = "modanet/datasets/coco/annotations/instances_all.json"
        self.num_class = 14
        self.model_dir = "coco_state_dict_model_v1.pt"
    
    def main(self):
        model = self.get_maskrcnn_model_for_train(self, self.num_class, hidden_layer =256)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model, all_train_losses, all_val_losses = self.train_model(model, train_dl, val_dl, num_epoch= 50)
        self.save_model(model)
    
    def save_model(self, model):
        # Specify a path
        torch.save(model.state_dict(), self.model_dir)
        print(f"{self.model_dir} saved")
    
    def _get_data(self):
        images = [i for i in sorted(os.listdir("modanet/datasets/coco/images")) if i.endswith("jpg") ]
        images  =list(set([i.get("file_name") for i in  annotations['images']]).intersection(set(images) ))
        return images
    
    def get_train_val_data(self, train_val_split = 0.9):
        images = self._get_data()
        
        num = int(train_val_split * len(images))
        num = num if num % 2 == 0 else num + 1
        
        train_imgs_inds = np.random.choice(range(len(images)) , num , replace = False)
        val_imgs_inds = np.setdiff1d(range(len(images)) , train_imgs_inds)
        train_imgs = np.array(images)[train_imgs_inds]
        val_imgs = np.array(images)[val_imgs_inds]
        
        c_train  = CustDat(train_imgs, annotation_dir = self.annotation_dir, root ="modanet/datasets/coco/" )
        c_val  = CustDat(val_imgs, annotation_dir= self.annotation_dir, root ="modanet/datasets/coco/" )
 
        return c_train, c_val

    def get_maskrcnn_model_for_inference(self):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights = "DEFAULT")
        model.eval()
        return model

    def get_maskrcnn_model_for_train(self, num_class, hidden_layer =256): 
        model = torchvision.models.detection.maskrcnn_resnet50_fpn()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features , num_class)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels    
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask , hidden_layer , num_class)
        return model 

    def train_model(self, model, train_dl, val_dl, num_epoch= 50):
        all_train_losses = []
        all_val_losses = []
        flag = False
        errors = []
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        
        for epoch in range(num_epoch ):
            try:
                train_epoch_loss = 0
                val_epoch_loss = 0
                model.train()
                for i , dt in enumerate(train_dl):
                    imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
                    targ = [dt[0][1] , dt[1][1]]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
                    loss = model(imgs , targets)
                    if not flag:
                        print(loss)
                        flag = True
                    losses = sum([l for l in loss.values()])
                    train_epoch_loss += losses.cpu().detach().numpy()
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                all_train_losses.append(train_epoch_loss)
                with torch.no_grad():
                    for j , dt in enumerate(val_dl):
                        imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
                        targ = [dt[0][1] , dt[1][1]]
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
                        loss = model(imgs , targets)
                        losses = sum([l for l in loss.values()])
                        val_epoch_loss += losses.cpu().detach().numpy()
                    all_val_losses.append(val_epoch_loss)
                print(epoch , "  " , train_epoch_loss , "  " , val_epoch_loss)
            except Exception as e :
                print(e)
                errors.append([imgs, targ])
                continue 

        return model,  all_train_losses, all_val_losses
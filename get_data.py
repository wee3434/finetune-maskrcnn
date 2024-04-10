import torchvision.transforms as T
from utils import annToMask
import json 

class CustDat(torch.utils.data.Dataset):
    def __init__(self, images, annotation_dir, root ="modanet/datasets/coco/" ):
        self.imgs = images
        self.annots =  json.loads(open(annotation_dir).read())
        self.root = root
    
    def __getitem__(self, idx):
        img = Image.open(self.root + "images/" + self.imgs[idx]).convert("RGB")
        target  = self._get_target( idx)
        return T.ToTensor()(img) , target
        
   
    def _get_target(self, idx):
        mask_lst ,box_lst, label_lst =  self._get_target_details( idx)
        target ={}
        target["boxes"] = box_lst
        target["masks"] = mask_lst
        target["labels"] = label_lst
        
        return target 
        
    def _get_target_details(self, idx):
        def get_id_from_file_name(file_name):
            return int(file_name[:-4])
        
        height, width = 0,0
        for i in  self.annots['images']:
            if i.get("file_name")==self.imgs[idx]:
                height , width = i.get("height"), i.get( "width") 
        
        if  height!=0 and width !=0 :
            annot_lst = []
            
            # extract all annotations of the given image
            for i in self.annots['annotations']:
                if i.get('image_id') == get_id_from_file_name(self.imgs[idx]):    
                    annot_lst.append(i) 
            
            num_objs = len(annot_lst) 
            masks = np.zeros((num_objs , height , width))
            labels =[]

            for the_id  in range(num_objs):
                ann = annot_lst[the_id]
                masks[the_id,:,:] = annToMask(ann, height , width)
                labels.append(ann['category_id'])  

            boxes = []
            for i in range(num_objs ):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin , ymin , xmax , ymax])
            boxes = torch.as_tensor(boxes , dtype = torch.float32)

            # covert to tensor type 
            mask_lst_tensor = torch.as_tensor(masks, dtype= torch.uint8) 
            box_lst_tensor = torch.as_tensor(boxes, dtype = torch.float32)
            label_encoded  =torch.tensor(labels, dtype=torch.int64)
            return mask_lst_tensor, box_lst_tensor, label_encoded
        
    def __len__(self):
        return len(self.imgs)
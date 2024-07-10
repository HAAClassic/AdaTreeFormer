import argparse
import torch
import os
import numpy as np
import datasets.crowd_gauss as crowd

import torch.nn.functional as F
from scipy.io import savemat 
from sklearn.metrics import r2_score
from scipy import spatial as ss


from network.Attention import QKV as ShotModel



parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='0', help='assign device')
parser.add_argument('--batch-size', type=int, default=1, help='train batch size') 
parser.add_argument('--crop-size', type=int, default=256, help='the crop size of the train image')
parser.add_argument('--model-path', type=str, default='.pth', help='saved model path')
parser.add_argument('--data-path', type=str, default='', help='dataset path')
parser.add_argument('--data-dir-ul', type=str, default='', help='data path')   
parser.add_argument('--dataset', type=str, default='TC')

def padding(array, values, axis):
    # This function should be doing post padding 0s.
    if axis not in {0,1}:
        print("Error! axis should be 0 or 1.")
        
    dim = array.shape
    new_dim = [0,0]
    for i in range(2):
        if i == axis:
            new_dim[i] = dim[i]+1
        else:
            new_dim[i] = dim[i]
    new_dim = tuple(new_dim)
    new_array = np.zeros(new_dim)
    
    for i in range(dim[0]):
        for j in range(dim[1]):
            new_array[i][j] = array[i][j]
    return new_array

def adjust_dim(array):
    # Make the dim even
    if array.shape[0]%2 != 0:
        array = padding(array, 0, 0)
    if array.shape[1]%2 != 0:
        array = padding(array, 0, 1)
    return array

def GAME_recursive(density, gt, currentLevel, targetLevel):
    if currentLevel == targetLevel:
        game = abs(np.sum(density) - np.sum(gt))
        return np.round(game, 3)
    
    else:
        density = adjust_dim(density)
        gt = adjust_dim(gt)
        density_slice = []; gt_slice = []
        
        density_slice.append(density[0:density.shape[0]//2, 0:density.shape[1]//2])
        density_slice.append(density[0:density.shape[0]//2, density.shape[1]//2:])
        density_slice.append(density[density.shape[0]//2:, 0:density.shape[1]//2])
        density_slice.append(density[density.shape[0]//2:, density.shape[1]//2:])

        gt_slice.append(gt[0:gt.shape[0]//2, 0:gt.shape[1]//2])
        gt_slice.append(gt[0:gt.shape[0]//2, gt.shape[1]//2:])
        gt_slice.append(gt[gt.shape[0]//2:, 0:gt.shape[1]//2])
        gt_slice.append(gt[gt.shape[0]//2:, gt.shape[1]//2:])
        
        currentLevel = currentLevel +1;
        res = []
        for a in range(4):
            res.append(GAME_recursive(density_slice[a], gt_slice[a], currentLevel, targetLevel))
        game = sum(res)
        return np.round(game, 3)

def GAME_metric(preds, gts, l):
    res = []
    for i in range(len(gts)):
        res.append(GAME_recursive(preds[i], gts[i], 0, l))
    return np.mean(res)

def hungarian(matrixTF):
    # matrix to adjacent matrix
    edges = np.argwhere(matrixTF)
    lnum, rnum = matrixTF.shape
    graph = [[] for _ in range(lnum)]
    for edge in edges:
        graph[edge[0]].append(edge[1])

    # deep first search
    match = [-1 for _ in range(rnum)]
    vis = [-1 for _ in range(rnum)]
    def dfs(u):
        for v in graph[u]:
            if vis[v]: continue
            vis[v] = True
            if match[v] == -1 or dfs(match[v]):
                match[v] = u
                return True
        return False

    # for loop
    ans = 0
    for a in range(lnum):
        for i in range(rnum): vis[i] = False
        if dfs(a): ans += 1

    # assignment matrix
    assign = np.zeros((lnum, rnum), dtype=bool)
    for i, m in enumerate(match):
        if m >= 0:
            assign[m, i] = True

    return ans, assign
  
def compute_metrics(dist_matrix,match_matrix,pred_num,gt_num,sigma,level):
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p,:]
        match_matrix[i_pred_p,:] = pred_dist<=sigma
        
    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0)==0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1)==1))[0]
    tp_gt_index = np.array(np.where(assign.sum(0)==1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1)==0))[0]

    tp = tp_pred_index.shape[0]
    fp = fp_pred_index.shape[0]
    fn = fn_gt_index.shape[0]

    tp_c = np.zeros(1)
    fn_c = np.zeros(1)
     

    return tp,fp,fn,tp_c,fn_c

def Gauss2Point(outputs):
    pred_map = F.avg_pool2d(outputs*100,3,1,1)
    maxm = F.max_pool2d(pred_map,3,1,1)
    maxm = torch.eq(maxm,pred_map)
    # import pdb;pdb.set_trace()
    pred_map = maxm*pred_map
    pred_map[pred_map<0.5]=0
    pred_map = pred_map.bool().long()
    pred_map = pred_map.cpu().data.numpy()[0,0,:,:] 
    return pred_map
    
def test(args, isSave = True):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda') 

    model_path = args.model_path
    crop_size = args.crop_size
    data_path = args.data_path
    data_dir_ul = args.data_dir_ul

    dataset = crowd.Crowd_TC(os.path.join(data_path, 'test_data'), crop_size, 1, method='val')
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, pin_memory=True)
    
    dataset_shot = crowd.Crowd_TC(os.path.join(data_dir_ul, 'train_data_01'), crop_size, 1, method='val')
    dataloader_shot = torch.utils.data.DataLoader(dataset_shot, 1, shuffle=False, num_workers=1, pin_memory=True)
    
        
    model = ShotModel('swin','swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth', False)
       
    
    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()
    image_errs = []
    result = []
    R2_es = []
    R2_gt = []
    tp_l, fp_l, fn_l, fn_c_l, tp_c_l = [], [], [], [], []
    l=0;
    for inputs, count, name, imgauss in dataloader:
        for inputs_shot, count_shot, name_shot, gausss_shot in dataloader_shot:
            break
        with torch.no_grad():
            inputs_shot = inputs_shot.to(device)
            imgauss = imgauss.to(device)
            inputs = inputs.to(device)
            mask_tree = ((inputs[:,1,:,:]-inputs[:,0,:,:])*(inputs[:,1,:,:]-inputs[:,2,:,:]))
                
            crop_imgs_shot, crop_masks_shot = [], []
            crop_gausss, crop_masks_gausss = [], []
            crop_imgs, crop_masks = [], []
            b, c, h, w = inputs.size()
            bsh, csh, hsh, wsh = inputs.size()
            rh, rw = args.crop_size, args.crop_size
            
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    randloc = abs(int(np.random.rand(1)*inputs_shot.shape[2])-rw)
                    crop_imgs_shot.append(inputs_shot[:, :, randloc:randloc+rh, randloc:randloc+rw])
                    crop_gausss.append(mask_tree[:, gis:gie, gjs:gje])
                    crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                    mask = torch.zeros([b, 1, h, w]).to(device)
                    mask_shot = torch.zeros([bsh, 1, hsh, wsh]).to(device)
                    mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                    crop_masks_gausss.append(mask)
                    crop_masks_shot.append(mask_shot)
                    crop_masks.append(mask)
                    
            crop_imgs_shot, crop_masks_shot = map(lambda x: torch.cat(x, dim=0), (crop_imgs_shot, crop_masks_shot))
            crop_gausss, crop_masks_gausss = map(lambda x: torch.cat(x, dim=0), (crop_gausss, crop_masks_gausss))
            crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))
                   
            
            crop_preds = []
            nz, bz = crop_imgs.size(0), args.batch_size
            for i in range(0, nz, bz):
                
                gs, gt = i, min(nz, i + bz)
                crop_pred,_,_ = model(crop_imgs[gs:gt],crop_imgs_shot[gs:gt])  
                crop_pred = crop_pred[0]
                
                _, _, h1, w1 = crop_pred.size()                
                crop_pred = F.interpolate(crop_pred, size=(h1 * 4, w1 * 4), mode='bilinear', align_corners=True) / 16
                crop_preds.append(crop_pred)
            crop_preds = torch.cat(crop_preds, dim=0)
            # splice them to the original size
            idx = 0
            pred_map = torch.zeros([b, 1, h, w]).to(device)
            for i in range(0, h, rh):
                gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                    pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                    idx += 1
            # for the overlapping area, compute average value
            mask = crop_masks.sum(dim=0).unsqueeze(0)
            outputs = pred_map / mask

            img_err = count[0].item() - torch.sum(outputs).item()
            R2_gt.append(count[0].item())
            R2_es.append(torch.sum(outputs).item())
            
            print("Img name: ", name, "Error: ", img_err, "GT count: ", count[0].item(), "Model out: ", torch.sum(outputs).item())
            image_errs.append(img_err)
            result.append([name, count[0].item(), torch.sum(outputs).item(), img_err])
            savemat('predictions/'+name[0]+'.mat', {'estimation':np.squeeze(outputs.cpu().data.numpy()),  'image': np.squeeze(inputs.cpu().data.numpy()), 'gt': np.squeeze(imgauss.cpu().data.numpy())})
            l=l+1
            
            
            
    image_errs = np.array(image_errs)
       
    ap_l = np.sum(tp_l)/(np.sum(tp_l)+np.sum(fp_l)+1e-20)
    ar_l = np.sum(tp_l)/(np.sum(tp_l)+np.sum(fn_l)+1e-20)
    f1m_l = 2*ap_l*ar_l/(ap_l+ar_l)
    ar_c_l = np.sum(tp_c_l)/(np.sum(tp_c_l)+np.sum(fn_c_l)+1e-20)
    
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    R_2 = r2_score(R2_gt,R2_es)

    print('{}: mae={}, mse={}, R2={}, pre={}, rec={}, f1={}\n'.format(model_path, mae, mse,R_2, ap_l, ar_l, f1m_l))

    if isSave:
        with open("test.txt","w") as f:
            for i in range(len(result)):
                f.write(str(result[i]).replace('[','').replace(']','').replace(',', ' ')+"\n")
            f.close()

if __name__ == '__main__':
    args = parser.parse_args()
    test(args, isSave= True)


"""
ALPNet
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alpmodule import MultiProtoAsConv
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder

#part aware prototype 
from .kmeans import KmeansClustering

# DEBUG
from pdb import set_trace

import pickle
import torchvision

# options for type of prototypes
FG_PROT_MODE = 'gridconv+' # using both local and global prototype
BG_PROT_MODE = 'gridconv'  # using local prototype only. Also 'mask' refers to using global prototype only (as done in vanilla PANet)

# thresholds for deciding class of prototypes
FG_THRESH = 0.95
BG_THRESH = 0.95

class FewShotSeg(nn.Module):
    """
    ALPNet
    Args:
        in_channels:        Number of input channels
        cfg:                Model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super(FewShotSeg, self).__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.get_encoder(in_channels)
        self.get_cls()

        #part aware prototypes
        self.global_const = 0.8
        self.kmeans = KmeansClustering(num_cnt=self.config['center'], iters=20, init='random')

    def get_encoder(self, in_channels):
        # if self.config['which_model'] == 'deeplab_res101':
        if self.config['which_model'] == 'dlfcn_res101':  #model used 
            use_coco_init = self.config['use_coco_init'] #pretrain on MS coco data (Microsoft Common Objects in Context)
            self.encoder = TVDeeplabRes101Encoder(use_coco_init) #feature extractor 

        else:
            raise NotImplementedError(f'Backbone network {self.config["which_model"]} not implemented')

        if self.pretrained_path:
            self.load_state_dict(torch.load(self.pretrained_path))
            print(f'###### Pre-trained model f{self.pretrained_path} has been loaded ######')

    def get_cls(self):
        """
        Obtain the similarity-based classifier
        """
        proto_hw = self.config["proto_grid_size"] #8
        feature_hw = self.config["feature_hw"]
        assert self.config['cls_name'] == 'grid_proto'
        if self.config['cls_name'] == 'grid_proto':
            self.cls_unit = MultiProtoAsConv(proto_grid = [proto_hw, proto_hw], feature_hw =  self.config["feature_hw"]) # when treating it as ordinary prototype
        else:
            raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')

    def getFeaturesArray(self, fts, mask, upscale=2):

        """
        Extract foreground and background features
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        c, h1, w1 = fts.shape[1:] #2048,53,53
        h, w = mask.shape[1:] #417,417

        fts1 = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True) #1,2048,417,417
        masked_fts = torch.sum(fts1 * mask[None, ...], dim=(2, 3)) \
                     / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C(2048) averaging across channels 

        mask_bilinear = F.interpolate(mask.unsqueeze(0), size=(h1*upscale, w1*upscale), mode='nearest').view(-1) #11236
        if mask_bilinear.sum(0) <= 10:
            fts = fts1.squeeze(0).permute(1, 2, 0).view(h * w, c)  ## l*c
            mask1 = mask.view(-1)
            if mask1.sum() == 0:
                fts = fts[[0]]*0  # 1 x C
            else:
                fts = fts[mask1>0]
        else:
            fts = F.interpolate(fts, size=(h1*upscale, w1*upscale), mode='bilinear', align_corners=True).squeeze(0).permute(1, 2, 0).view(h1*w1*upscale**2, c)
            fts = fts[mask_bilinear>0] #2157, 2048
        return (fts, masked_fts) #local, global
    
    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype

    def kmeansPrototype(self, fg_fts, bg_fts):

        fg_fts_loc = [torch.cat([tr[0] for tr in way], dim=0) for way in fg_fts] ## concat all fg_fts 1,3080,2048
        fg_fts_glo = [[tr[1] for tr in way] for way in fg_fts]  ## all global 1, 1, 2048
        bg_fts_loc = torch.cat([torch.cat([tr[0] for tr in way], dim=0) for way in bg_fts], dim=0)
        bg_fts_glo = [[tr[1] for tr in way] for way in bg_fts] #1,1,1,2048
        fg_prop_cls = [self.kmeans.cluster(way) for way in fg_fts_loc]
        bg_prop_cls = self.kmeans.cluster(bg_fts_loc)

        fg_prop_glo, bg_prop_glo = self.getPrototype(fg_fts_glo, bg_fts_glo)
        fg_propotypes = [fg_c + self.global_const*fg_g for (fg_c, fg_g) in zip(fg_prop_cls, fg_prop_glo)]
        bg_propotypes = bg_prop_cls + self.global_const * bg_prop_glo

        return fg_propotypes, bg_propotypes, fg_prop_glo, bg_prop_glo

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz = False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            show_viz: return the visualization dictionary
        """
        # ('Please go through this piece of code carefully')
        n_ways = len(supp_imgs) #no of classes 
        n_shots = len(supp_imgs[0]) #no of labelled images in each class 
        n_queries = len(qry_imgs)

        assert n_ways == 1, "Multi-shot has not been implemented yet" # NOTE: actual shot in support goes in batch dimension
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:] #height?
        qry_bsize = qry_imgs[0].shape[0]

        assert sup_bsize == qry_bsize == 1

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] #stack by ways #support images
                                + [torch.cat(qry_imgs, dim=0),], dim=0) #query images

        img_fts = self.encoder(imgs_concat, low_level = False) #low level False: do not return aggregated low level features in FCN
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad = True) #for calculating gradients
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        visualizes = [] # the buffer for visualization

        for epi in range(1): # batch dimension, fixed to 1
            fg_masks = [] # keep the way part

            '''
            for way in range(n_ways):
                # note: index of n_ways starts from 0
                mean_sup_ft = supp_fts[way].mean(dim = 0) # [ nb, C, H, W]. Just assume batch size is 1 as pytorch only allows this
                mean_sup_msk = F.interpolate(fore_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear')
                fg_masks.append( mean_sup_msk )

                mean_bg_msk = F.interpolate(back_mask[way].mean(dim = 0).unsqueeze(1), size = mean_sup_ft.shape[-2:], mode = 'bilinear') # [nb, C, H, W]
            '''
            # re-interpolate support mask to the same size as support feature
            res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size = fts_size, mode = 'bilinear') for fore_mask_w in fore_mask], dim = 0) # [nway, ns, nb, nh', nw']
            res_bg_msk = torch.stack([F.interpolate(back_mask_w, size = fts_size, mode = 'bilinear') for back_mask_w in back_mask], dim = 0) # [nway, ns, nb, nh', nw']
            #F.interpolate is for downsampling/upsampling input 

            #####################################generate part aware prototypes###################################

            supp_fg_fts = [[self.getFeaturesArray(supp_fts[way, shot, [epi]], res_fg_msk[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)] #extract foreground and background features
            supp_bg_fts = [[self.getFeaturesArray(supp_fts[way, shot, [epi]], res_bg_msk[way, shot, [epi]], 1)
                            for shot in range(n_shots)] for way in range(n_ways)]
            fg_prototypes, bg_prototype, fg_fts, bg_fts = self.kmeansPrototype(supp_fg_fts, supp_bg_fts)

            #####################################generate part aware prototypes###################################

            scores          = []
            assign_maps     = []
            bg_sim_maps     = []
            fg_sim_maps     = []
            #calculating similar scores here?
            _raw_score, _, aux_attr = self.cls_unit(qry_fts, supp_fts, res_bg_msk, mode = BG_PROT_MODE, thresh = BG_THRESH, isval = isval, val_wsize = val_wsize, vis_sim = show_viz  ) #prototype
            #bg_prot_mode: using local prototype
            scores.append(_raw_score)
            assign_maps.append(aux_attr['proto_assign'])
            if show_viz:
                bg_sim_maps.append(aux_attr['raw_local_sims'])
            
            for way, _msk in enumerate(res_fg_msk):
                _raw_score, _, aux_attr = self.cls_unit(qry_fts, supp_fts, _msk.unsqueeze(0), partProto=fg_prototypes[way], mode = FG_PROT_MODE if F.avg_pool2d(_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', thresh = FG_THRESH, isval = isval, val_wsize = val_wsize, vis_sim = show_viz  )

                scores.append(_raw_score)
                if show_viz:
                    fg_sim_maps.append(aux_attr['raw_local_sims'])

            pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        assign_maps = torch.stack(assign_maps, dim = 1)
        bg_sim_maps    = torch.stack(bg_sim_maps, dim = 1) if show_viz else None
        fg_sim_maps    = torch.stack(fg_sim_maps, dim = 1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps


    # Batch was at the outer loop
    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Masks for getting query prototype
        pred_mask = pred.argmax(dim=1).unsqueeze(0)  #1 x  N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]

        # skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        # FIXME: fix this in future we here make a stronger assumption that a positive class must be there to avoid undersegmentation/ lazyness
        skip_ways = []

        ### added for matching dimensions to the new data format
        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2) # added to nway(1) and nb(1)

        ### end of added part

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                img_fts = supp_fts[way: way + 1, shot: shot + 1] # actual local query [way(1), nb(1, nb is now nshot), nc, h, w]

                qry_pred_fg_msk = F.interpolate(binary_masks[way + 1].float(), size = img_fts.shape[-2:], mode = 'bilinear') # [1 (way), n (shot), h, w]

                # background
                qry_pred_bg_msk = F.interpolate(binary_masks[0].float(), size = img_fts.shape[-2:], mode = 'bilinear') # 1, n, h ,w
                
                #generate part aware prototypes 
                qry_fg_fts = [[self.getFeaturesArray(img_fts[way, [shot]], qry_pred_fg_msk[way, [shot]])]] #extract foreground and background features
                qry_bg_fts = [[self.getFeaturesArray(img_fts[way, [shot]], qry_pred_bg_msk[way, [shot]], 1)]]
                qry_fg_prototypes, qry_bg_prototype, qry_fg_fts, qry_bg_fts = self.kmeansPrototype(qry_fg_fts, qry_bg_fts)

                scores = []

                _raw_score_bg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_bg_msk.unsqueeze(-3), mode = BG_PROT_MODE, thresh = BG_THRESH )

                scores.append(_raw_score_bg)

                _raw_score_fg, _, _ = self.cls_unit(qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_fg_msk.unsqueeze(-3), partProto=qry_fg_prototypes[way], mode = FG_PROT_MODE if F.avg_pool2d(qry_pred_fg_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', thresh = FG_THRESH )
                scores.append(_raw_score_fg)

                supp_pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear')

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss.append( F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways)

        return torch.sum( torch.stack(loss))

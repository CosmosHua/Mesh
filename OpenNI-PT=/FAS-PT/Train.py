# coding:utf-8
# !/usr/bin/python3

from Loader import *
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#from Net_Shuffle import ShuffleNet as model
#from Net_Squeeze import SqueezeNet2 as model
from Net_Feather import FeatherNetB as model


from time import time
from torch import nn, optim; import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################################################################################
def TrainModel(dir, dst, val, Epoch=100, bs=64, wp=5):
    Loader, cls = LoadData(dir, train=True, bs=bs)
    net = model(inc,cls).to(device); net.train() # train_mode
    
    wt0 = [len(os.listdir(dir+"/"+i)) for i in cls.values()]
    wt0 = (1-torch.Tensor(wt0)/sum(wt0)).to(device) # global_weight
    #LF = nn.CrossEntropyLoss(weight=wt0) # global balanced CEL
    LF = FocalLoss(wt0) # global_weight Focal_Loss
    
    optmz = optim.SGD(net.parameters(), lr=0.1*(bs/256), momentum=0.9,
            weight_decay=1E-4, dampening=0, nesterov=True) # better>adam
    # (factor,patience,cooldown): (0.8,8,3), (0.5,8,3), (0.5,3,3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optmz, mode='min',
                factor=0.8, patience=8, verbose=True, cooldown=3) # loss
    
    N = min(len(Loader)//2, max(100,Epoch)); K = len(cls)
    sk = LoadParam(dst, net, optmz) if os.path.isfile(dst) else 0
    if sk>0: dst = dst.replace("_%d"%sk, "") # rename for saving
    try: # in case: out of memeory
        for ep in range(0, Epoch):
            ep += sk+1; loss, acc = 0, 0
            for n, batch in enumerate(Loader, 1):
                x, y = [b.to(device) for b in batch] # inputs
                #x,y = [Variable(b).to(device) for b in batch]
                ym = net(x) # forward->(batch,cls)
                
                #wt = torch.Tensor([sum(y==i) for i in range(K)])
                #wt = (1-wt/sum(wt)).to(device) # batch_weight
                #LF = nn.CrossEntropyLoss(wt) # batch balanced CEL
                #LF = FocalLoss(wt) # batch_weight Focal_Loss
                L = LF(ym,y) # global/batch_weight Focal_Loss/CEL

                # zero_grad() must before backward()
                optmz.zero_grad() # clear gradients
                L.backward() # backward->get gradients
                optmz.step() # update parameters
                
                loss += L.item() # tensor->number: float(L)
                v, k = torch.max(ym, dim=1) # (value,index)
                acc += (y==k).double().mean().item() # mean->number
                #if n%N==0: print("[Ep:%03d, Ba:%03d] Loss: %.3f, Acc: %.3f"
                #    % (ep,n,loss/n,acc/n) ) # N-batch accumulated statistics
            
            loss /= n; acc /= n # per-image average
            loss_v, acc_v, t = EvalModel(val, net.state_dict(), bs)
            print("[Ep:%03d] Loss: %.3f|%.3f, Acc: %.3f|%.3f, T=%.1fms."
                    % (ep, loss, loss_v, acc, acc_v, t) )
            
            SavePolicy(dst, net, ep, acc*acc_v, optmz) # save param
            if acc>0.99 and acc_v>0.98: break # return # early stop
            if scheduler: scheduler.step(loss) # min->loss, max->acc
    
    finally: # in case: max_epoch, early_stop, any training error
        return SavePolicy(dst, net, ep, acc_v, optmz) # save


def EvalModel(dir, param, bs=64): # with labels
    Loader, cls = LoadData(dir, train=False, bs=bs)
    net = model(inc,cls).to(device); net.eval() # eval_mode
    
    wt = [len(os.listdir(dir+"/"+i)) for i in cls.values()]
    wt = (1-torch.Tensor(wt)/sum(wt)).to(device) # global_weight
    LF = nn.CrossEntropyLoss(weight=wt) # mean balanced CEL
    
    if type(param)!=str: net.load_state_dict(param) # state_dict
    elif os.path.isfile(param): LoadParam(param, net) # file
    
    t0 = time(); loss, acc = 0, 0
    with torch.no_grad(): # inference
        if bs<25: print(cls) # show {index: class}
        for n, (x,y) in enumerate(Loader, 1):
            x,y = [b.to(device) for b in (x,y)] # inputs
            #x,y = [Variable(b).to(device) for b in (x,y)]
            ym = net(x) # forward->(batch,cls)
            loss += LF(ym,y).item() # Tensor->number
            v, k = torch.max(ym, dim=1) # (value,index)
            acc += (y==k).double().mean().item() # mean->number
            if bs<25: print("Guess:",k.cpu(), "\nLabel:",y.cpu())
    loss /= n; acc /= n; t0 = (time()-t0)/n # per-batch time
    return loss, acc, t0*1000 # ms


SQ = {} # Ref: https://arxiv.org/abs/1812.01187
################################################################################
def SavePolicy(dst, net, ep, acc, opt=None, Num=6):
    mi = min(SQ.values()) if len(SQ)>0 else 0.9
    if acc<mi or ep in SQ: return # acc>=min>=0.9
    
    out = "[ep=%03d]:%.5f "%(ep,acc)
    if len(SQ)==Num: # replace first lowest
        k = [k for k in SQ if SQ[k]==mi][0] # first
        out += "<-[%03d]:%.5f"%(k,mi); SQ.pop(k)
        os.remove(dst.replace(".pth", "_%d.pth"%k))
    with open("sq.log", "a+") as f: f.write(out+"\n")
    SQ[ep] = acc; return SaveParam(dst, net, ep, opt)


def SaveParam(dst, net, ep, opt=None): # save ckpt
    ckpt = {"model": net.state_dict(), "epoch": ep}
    #ckpt = {"model": net.cpu().state_dict(), "epoch": ep}
    if dst[-4:]==".tar" and opt: ckpt["optim"] = opt.state_dict()
    dst = dst.replace(".pth", "_%d.pth"%ep); torch.save(ckpt, dst)
    print("=>Save Ckpt: %s"%dst); return ckpt # *.pth/.pth.tar


def LoadParam(dst, net, opt=None): # load ckpt
    dt = time(); assert os.path.isfile(dst) # check file
    ckpt = torch.load(dst, map_location=device) # *.pth/.pth.tar
    if dst[-4:]==".tar" and opt: opt.load_state_dict(ckpt["optim"])
    net.load_state_dict(ckpt["model"]); dt = time()-dt # load param
    print("=>Load Ckpt: %s\t%.3fs."%(dst,dt)); return ckpt["epoch"]


def WarmUp(optmz, i, N):
    optmz.param_groups[0]["lr"] = optmz.defaults["lr"]*(i/N)


################################################################################
class FocalLoss(nn.Module):
    """Loss(X,k) = -weight(k) * (1-softmax(X,k))^gamma * log(softmax(X,k)).
    Default: The losses are averaged across observations for each minibatch.
    # Ref: https://arxiv.org/abs/1708.02002
    Args:
        weight(1D tensor|class_num): the weighting factor for class imbalance.
        gamma(float>0): reduces the relative loss for well-classiﬁed examples(p>0.5),
            in order to put more focus on hard, misclassiﬁed examples. Default: 2.
        reduction(string): the reduction to apply to the output: 'none'|'mean'|'sum'.
            'none': no reduction will be applied; 'sum': the output will be summed;
            'mean': the mean value of the outputs. Default: 'mean'."""
    def __init__(self, weight, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma; self.rd = reduction
        assert isinstance(weight, (int, torch.Tensor))
        self.wt = torch.ones(weight) if type(weight)==int else weight

    def forward(self, input, target):
        """Args: input (2D Tensor): (N=batch_size, C=class_num), probs.
                 target (1D Tensor): (N=batch_size), labels.
        Returns: the focal loss of input/inference and taget/label."""
        wt = self.wt.to(input.device)
        p = torch.softmax(input, dim=1)
        loss = -wt * (1-p)**self.gamma * p.log()
        
        # Method_1:
        mask = torch.zeros_like(loss)
        mask.scatter_(1, target.view(-1,1), 1.0) # dim,index,src
        loss = (loss*mask).sum(dim=1)
        # Method_2:
        #loss = [loss[i,k] for i,k in enumerate(target)]
        #loss = torch.tensor(loss, device=wt.device)
        
        if self.rd=="mean": loss = loss.mean()
        elif self.rd=="sum": loss = loss.sum()
        return loss # also for "none"


################################################################################
if __name__ == "__main__":
    train = "../train"; val = "../val"
    tp = ".pth"; pre = "RGBD%d" % cmb
    ep = 600; para = pre + "_%d"%ep + tp
    
    TrainModel(train, pre+tp, val, ep) # from init
    #TrainModel(train, para, val, ep) # from ckpt
    #EvalModel(train, para) # on training set
    #EvalModel(val, para, 20) # on validation


################################################################################

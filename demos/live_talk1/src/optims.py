import numpy as np
# from . import sls
from . import others
from . import adaptive_first, adaptive_second
import torch
from src.optimizers import sls, sps, ssn


def get_optimizer(opt, params, n_batches_per_epoch=None, n_train=None, lr=None,
                  train_loader=None, model=None, loss_function=None, exp_dict=None, batch_size=None):
    """
    opt: name or dict
    params: model parameters
    n_batches_per_epoch: b/n
    """
    if isinstance(opt, dict):
        opt_name = opt["name"]
        opt_dict = opt
    else:
        opt_name = opt
        opt_dict = {}

    # ===============================================
    # our optimizers   
    n_batches_per_epoch = opt_dict.get("n_batches_per_epoch") or n_batches_per_epoch    
    

    if opt_name == "adaptive_second":

        opt = adaptive_second.AdaptiveSecond(params,
                    c = opt_dict.get('c', None),
                    n_batches_per_epoch=n_batches_per_epoch,
                    # gv_option=opt_dict.get('gv_option', 'per_param'),
                    base_opt=opt_dict.get('base_opt', 'adagrad'),
                    accum_gv=opt_dict.get('accum_gv', None),
                    lm=opt_dict.get('lm', 0),
                    avg_window=opt_dict.get('window', 10),
                    pp_norm_method=opt_dict.get('pp_norm_method', 'pp_armijo'),
                    momentum=opt_dict.get('momentum', 0),
                    beta=opt_dict.get('beta', 0.99),
                    gamma=opt_dict.get('gamma', 2),
                    # apply_sqrt=opt_dict.get('apply_sqrt', True),
                    init_step_size=opt_dict.get('init_step_size', 1),
                    adapt_flag=opt_dict.get('adapt_flag', 'constant'), 
                    step_size_method=opt_dict.get('step_size_method', 'sls'), 
                    # sls stuff
                    beta_b=opt_dict.get('beta_b', .9),
                    beta_f=opt_dict.get('beta_f', 2.),
                    reset_option=opt_dict.get('reset_option', 1),
                    line_search_fn=opt_dict.get('line_search_fn', "armijo"),   
                    )
                    
    elif opt_name == "adaptive_first":

        opt = adaptive_first.AdaptiveFirst(params,
                    c = opt_dict['c'],
                    n_batches_per_epoch=n_batches_per_epoch,
                    gv_option=opt_dict.get('gv_option', 'per_param'),
                    base_opt=opt_dict['base_opt'],
                    pp_norm_method=opt_dict['pp_norm_method'],
                    momentum=opt_dict.get('momentum', 0),
                    beta=opt_dict.get('beta', 0.99),
                    gamma=opt_dict.get('gamma', 2),
                    init_step_size=opt_dict.get('init_step_size', 1),
                    adapt_flag=opt_dict.get('adapt_flag', 'constant'), 
                    step_size_method=opt_dict['step_size_method'], 
                    # sls stuff
                    beta_b=opt_dict.get('beta_b', .9),
                    beta_f=opt_dict.get('beta_f', 2.),
                    reset_option=opt_dict.get('reset_option', 1),
                    line_search_fn=opt_dict.get('line_search_fn', "armijo"),   
                    )
   
    elif opt_name == "sgd_armijo":
        # if opt_dict.get("infer_c"):
        #     c = (1e-3) * np.sqrt(n_batches_per_epoch)
        if opt_dict['c'] == 'theory':
            c = (n_train - batch_size) / (2 * batch_size * (n_train - 1))
        else:
            c = opt_dict.get("c") or 0.1
        
        opt = sls.Sls(params,
                    c = c,
                    n_batches_per_epoch=n_batches_per_epoch,
                    init_step_size=opt_dict.get("init_step_size", 1),
                    line_search_fn=opt_dict.get("line_search_fn", "armijo"), 
                    gamma=opt_dict.get("gamma", 2.0),
                    reset_option=opt_dict.get("reset_option", 1),
                    eta_max=opt_dict.get("eta_max"))


    elif opt_name == "sgd_goldstein":
        opt = sls.Sls(params, 
                      c=opt_dict.get("c") or 0.1,
                      reset_option=opt_dict.get("reset_option") or 0,
                      n_batches_per_epoch=n_batches_per_epoch,
                      line_search_fn="goldstein")

    elif opt_name == "sgd_nesterov":
        opt = sls.SlsAcc(params, 
                        acceleration_method="nesterov", 
                        gamma=opt_dict.get("gamma", 2.0),
                        aistats_eta_bound=opt_dict.get("aistats_eta_bound", 10.0))

    elif opt_name == "sgd_polyak":
        opt = sls.SlsAcc(params, 
                         c=opt_dict.get("c") or 0.1,
                         momentum=opt_dict.get("momentum", 0.6),
                         n_batches_per_epoch=n_batches_per_epoch,
                         gamma=opt_dict.get("gamma", 2.0),
                         acceleration_method="polyak",
                         aistats_eta_bound=opt_dict.get("aistats_eta_bound", 10.0),
                         reset_option=opt_dict.get("reset", 0))

    elif opt_name == "seg":
        opt = sls.SlsEg(params, n_batches_per_epoch=n_batches_per_epoch)

    elif opt_name == "ssn":
        opt = ssn.Ssn(params, 
            n_batches_per_epoch=n_batches_per_epoch, 
            init_step_size=opt_dict.get('init_step_size', 1.0), 
            lr=None, 
            c=opt_dict.get('c',0.1), 
            beta=0.9, 
            gamma=1.5,
            reset_option=1, 
            lm=opt_dict.get("lm", 0))

    # ===============================================
    # others
    elif opt_name == "adam":
        opt = torch.optim.Adam(params, amsgrad=opt.get('amsgrad'),  lr=opt['lr'],  betas=opt.get('betas', (0.9,0.99)))

    elif opt_name == "adagrad":
        opt = torch.optim.Adagrad(params, lr=opt['lr'])

    elif opt_name == 'sgd':
        # best_lr = lr if lr else 1e-3
        opt = torch.optim.SGD(params, lr=opt['lr'])

    elif opt_name == 'rmsprop':
        opt = torch.optim.RMSprop(params, lr=opt['lr'])

    elif opt_name == 'sps':
        opt = sps.Sps(params, c=.2, 
                        n_batches_per_epoch=n_batches_per_epoch, 
                        adapt_flag=opt_dict.get('adapt_flag', 'basic'),
                        fstar_flag=opt_dict.get('fstar_flag'),
                        eta_max=opt_dict.get('eta_max'),
                        eps=opt_dict.get('eps', 0))
    else:
        raise ValueError("opt %s does not exist..." % opt_name)

    return opt


class Sps(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.1,
                 gamma=2.0,
                 eta_max=10,
                 adapt_flag=None,
                 fstar_flag=None,
                 eps=0):
        params = list(params)
        super().__init__(params, {})
        self.eps = eps
        self.params = params
        self.c = c
        self.eta_max = eta_max
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.adapt_flag = adapt_flag
        self.state['step'] = 0
        self.state['step_size_avg'] = 0.

        self.state['step_size'] = init_step_size
        self.step_size_max = 0.
        self.n_batches_per_epoch = n_batches_per_epoch

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.state['lb'] = None
        self.loss_min = np.inf
        self.loss_sum = 0.
        self.loss_max = 0.
        self.fstar_flag = fstar_flag

    def step(self, closure, batch, clip_grad=False):
        indices = batch['meta']['indices']
        # deterministic closure
        seed = time.time()

        batch_step_size = self.state['step_size']

        if self.fstar_flag:
            fstar = float(batch['meta']['fstar'].mean())
        else:
            fstar = 0.

        # get loss and compute gradients
        loss = closure()
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(self.params, 0.25)

        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1

        if self.state['step'] % int(self.n_batches_per_epoch) == 0:
            self.state['step_size_avg'] = 0.

        self.state['step'] += 1

        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = ut.get_grad_list(self.params)

        grad_norm = ut.compute_grad_norm(grad_current)
        
        if grad_norm < 1e-6:
            return 0.

        if self.adapt_flag == 'moving_lb':
            if (self.state['lb'] is not None) and loss > self.state['lb']: 
                step_size = (loss - self.state['lb']) / (self.c * (grad_norm)**2)
                step_size = step_size.item()
                loss_scalar = loss.item()
            else:
                step_size = 0.
                loss_scalar = loss.item()

            if (self.state['step'] % int(self.n_batches_per_epoch)) == 0:
                # do update on lower bound 
                # new_bound = self.loss_sum / self.n_batches_per_epoch
                # self.state['lb'] = (self.loss_sum / self.n_batches_per_epoch) / 100.
                # self.state['lb'] /= 2.
                # self.state['lb'] = self.loss_max / 1000.
                self.state['lb'] = self.loss_min / 100.
                # if new_bound > self.state['lb']:
                #     self.state['lb'] = (new_bound - self.state['lb']) / 2.
                # self.loss_sum = 0.
                print('lower_bound:', self.state['lb'])

            self.loss_sum += loss_scalar
            self.loss_min = min(loss_scalar, self.loss_min)
            self.loss_max = max(loss_scalar, self.loss_max)

        elif self.adapt_flag in ['constant']:
            step_size = (loss - fstar) / (self.c * (grad_norm)**2 + self.eps)
            if loss < fstar:
                step_size = 0.
                loss_scalar = 0.
            else:
                if self.eta_max is None:
                    step_size = step_size.item()
                else:
                    step_size =  min(self.eta_max, step_size.item())
                    
                loss_scalar = loss.item()

        elif self.adapt_flag in ['basic']:
            if torch.isnan(loss):
                raise ValueError('loss is NaN')
            # assert(loss >= fstar)
            if loss < fstar:
                step_size = 0.
                loss_scalar = 0.
            else:
                step_size = (loss - fstar) / (self.c * (grad_norm)**2+ self.eps)
                step_size =  step_size.item()

                loss_scalar = loss.item()

        elif self.adapt_flag in ['smooth_iter']:
            step_size = loss / (self.c * (grad_norm)**2)
            coeff = self.gamma**(1./self.n_batches_per_epoch)
            step_size =  min(coeff * self.state['step_size'], 
                             step_size.item())
           
            loss_scalar = loss.item()

        elif self.adapt_flag == 'smooth_epoch':
            step_size = loss / (self.c * (grad_norm)**2)
            step_size = step_size.item()
            if self.state['step_size_epoch'] != 0:
                step_size =  min(self.state['step_size_epoch'], 
                                 step_size)
                self.step_size_max = max(self.step_size_max, step_size)
            else:
                self.step_size_max = max(self.step_size_max, step_size)
                step_size = 0.
                

            loss_scalar = loss.item()

             # epoch done
            if (self.state['step'] % int(self.n_batches_per_epoch)) == 0:
                self.state['step_size_epoch'] = self.step_size_max
                self.step_size_max = 0.

        try_sgd_update(self.params, step_size, params_current, grad_current)

        # save the new step-size
        self.state['step_size'] = step_size

        
        self.state['step_size_avg'] += (step_size / self.n_batches_per_epoch)
        self.state['grad_norm'] = grad_norm.item()
        
        if torch.isnan(self.params[0]).sum() > 0:
            print('nan')

        return loss


def try_sgd_update(params, step_size, params_current, grad_current):
    zipped = zip(params, params_current, grad_current)

    for p_next, p_current, g_current in zipped:
        if g_current is None:
            continue
        p_next.data[:] = p_current.data
        p_next.data.add_(- step_size, g_current)

def compute_grad_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


def get_grad_list(params):
    g_list = []
    for p in params:
        grad = p.grad
        if grad is None:
            grad = 0.

        g_list += [grad]
        
    return g_list

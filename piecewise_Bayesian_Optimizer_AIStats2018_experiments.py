
#Using something like particle-filter for Bayesian optimization

import numpy as np
import nlopt
from scipy.stats import norm
import random
import sklearn.gaussian_process as gp
import GPy
SIGMA = 1

class node:
    def __init__(self,model,x_array,y_array,min_threshold,max_threshold,quality,best_threshold_index,best_threshold_cutoff,mean,variance):
        """
        :param model: fitted-model
        :param x_array: parameter-values for the current-node
        :param y_array: noisy samples for the current-node
        :param min_threshold: min-threshold value
        :param max_threshold: max-threshold value
        :param quality: quality of the current node
        :param best_threshold_index: the threshold index assigned to the deciding feature
        :param best_threshold_cutoff: the threshold value
        :param mean:
        :param variance:
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.quality = quality
        self.x_array = x_array
        self.y_array = y_array
        self.model = model

        self.best_threshold_index = best_threshold_index
        self.best_threshold_cutoff = best_threshold_cutoff

        self.mean = mean
        self.variance = variance

        self.left_child = None
        self.right_child = None

    def set_left_child(self,left_node):
        self.left_child = left_node

    def set_right_child(self,right_node):
        self.right_child = right_node


def inference(input_record,root_node):
    """
    :Running inference for piece-wise GP
    :param input_record: input parameter value
    :param root_node: Root-node of the constructed tree
    :return: prediction and assigned variance
    """
    node_index = 0
    current_node = root_node
    condition  = True
    while condition:
        best_threshold_index = current_node.best_threshold_index
        threshold_cutoff = current_node.best_threshold_cutoff
        if not threshold_cutoff: break
        if input_record[best_threshold_index]<threshold_cutoff:
            #leftNode
            current_node = current_node.left_child
        else:
            #rightNode
            current_node = current_node.right_child
    model = current_node.model
    prediction,variance = model.predict(np.array([list(input_record)]))
    return (prediction,variance)

def sample_cost_function(X,power):
    """
    :generate noisy samples for the cost function
    :param X: input parameter
    :param power:
    :return: function value
    """
    noise = np.random.normal(0, (.01 ** 2))
    x = X[0]
    return ([x**power+noise])

def sample_function(X,std):
    """
    :Generate noisy samples for the constraint
    :param X: parameter value
    :param std: standard deviation of additive noise
    :return: noisy generate value
    """

    noise = np.random.normal(0, (std ** 2))

    x = X[0]
    if x>0:
        return ([1],[1+noise])
    else:
        return ([-1],[-1+noise])


def eval_node(x_array,y_array,threshold,threshold_index):

    """
    :Evaluates the quality of the current node
    :param x_array: all parameter values assigned to this node
    :param y_array: all noisy samples assigned to thi node
    :param threshold: threshold-value
    :param threshold_index: threshold-index (multi-dimensional case)
    :return:
    """
    new_x_array_right = []
    new_y_array_right = []
    new_x_array_left = []
    new_y_array_left = []

    for index,x in enumerate(x_array):
        if x[threshold_index]>threshold:
            new_x_array_right.append(x)
            new_y_array_right.append(y_array[index])
        else:
            new_x_array_left.append(x)
            new_y_array_left.append(y_array[index])

    new_x_array_right = np.array(new_x_array_right)
    new_x_array_left = np.array(new_x_array_left)
    new_y_array_right = np.array(new_y_array_right)
    new_y_array_left = np.array(new_y_array_left)


    model_left = GPy.models.GPRegression(new_x_array_left, new_y_array_left)
    model_left.optimize("bfgs")

    model_right = GPy.models.GPRegression(new_x_array_right,new_y_array_right)
    model_right.optimize("bfgs")

    mean_right,var_right = model_right.predict(new_x_array_right)
    mean_left, var_left = model_left.predict(new_x_array_left)

    return (model_left,model_right,new_x_array_right,new_y_array_right,new_x_array_left,
            new_y_array_left,np.mean(var_right),np.mean(var_left),np.var(var_right),np.var(var_left))

def build_gp_tree(node_vector,threshold_dim):

    """
    :This builds a piece-wise Gaussian Regression
    :param node_vector: list of all created nodes
    :param threshold_dim: dimension of the parameter-space
    :return:
    """

    vec_list = list(node_vector)
    mu = .9
    while not not vec_list:
        mu+= .1
        parent =vec_list[0]
        vec_list = vec_list[1:]

        #print(parent.min_threshold, parent.max_threshold)
        x_array = parent.x_array
        y_array = parent.y_array

        quality = 100000
        var = 1

        left = None
        right = None
        left_x = None
        left_y = None
        right_x = None
        right_y = None
        left_qual = None
        right_qual = None
        optimal_th = None
        best_threshold_index = None
        #Loop over parameters (thresholds)
        for threshold_index in range(0,threshold_dim):

            #Loop over threshold values
            min_val = parent.min_threshold[threshold_index]
            max_val = parent.max_threshold[threshold_index]
            for th in np.arange(min_val,max_val,.1):
                try:
                    model_left,model_right,x_right,y_right,x_left,y_left,qual_right,qual_left,var_right,var_left = \
                        eval_node(x_array, y_array, th,threshold_index)
                except:
                    #print("ERROR")
                    continue

                #evaluate the quality
                node_mean_qual = (qual_right+qual_left)/2

                left_size = len(x_left)
                right_size = len(x_right)
                if left_size<20 or right_size<20: continue

                #node_mean_qual = (left_size*qual_left+right_size*qual_right)/(left_size+right_size) #weighted mean
                node_mean_qual = ((left_size*qual_left+right_size*qual_right)/(left_size+right_size))
                node_var_qual = ((left_size * np.sqrt(var_left) + right_size * np.sqrt(var_right)) / (left_size + right_size))**2
                #print(th, node_mean_qual)
                if node_mean_qual < quality:
                    #quality = node_qual
                    quality = node_mean_qual
                    var = node_var_qual
                    optimal_th = th
                    left = model_left
                    right = model_right
                    left_x = x_left
                    left_y = y_left
                    right_x = x_right
                    right_y = y_right
                    left_qual = qual_left
                    right_qual = qual_right
                    best_threshold_index = threshold_index


        #print(best_threshold_index,optimal_th)
        #print(best_threshold_index)
        if not optimal_th:continue
        #print(optimal_th,parent.quality/quality)
        p_hat = (parent.quality+quality)/2.0
        difference = (parent.quality-quality)/np.sqrt(p_hat*(1-p_hat)*(2.0/(left_size+right_size)))
        #print(difference)
        num_left_samples = len(left_x)
        num_right_samples = len(right_x)

        print(parent.quality/quality,mu)
        if parent.quality / quality > mu:
        #if norm.cdf(difference)>.6 and num_left_samples>10 and num_right_samples>10:

            #create min/max threshold values
            right_min_threshold = list(parent.min_threshold)
            right_min_threshold[best_threshold_index] = optimal_th

            left_max_threshold = list(parent.max_threshold)
            left_max_threshold[best_threshold_index] = optimal_th

            node_left = node(left,left_x,left_y,parent.min_threshold,left_max_threshold,left_qual,best_threshold_index,None,left_qual,var)
            node_right = node(right,right_x,right_y,right_min_threshold,parent.max_threshold,right_qual,best_threshold_index,None,right_qual,var)
            node_vector.append(node_left)
            node_vector.append(node_right)
            #change these parts for multi-dimensional case
            parent.best_threshold_index = best_threshold_index
            parent.best_threshold_cutoff = optimal_th
            parent.set_left_child(node_left)
            parent.set_right_child(node_right)
            vec_list.append(node_left)
            vec_list.append(node_right)

def acquisition(x,grad,gp_model,nodes,min_val,temperature,type):

    """
    :calculate expected improvement
    :param gp_model: trained Gaussian-process on the loss function
    :param gp_model_constraint1: trained Gaussian-process on the constraint values
    :param min_val: minimum value of the loss function (so-far)
    :param x:
    :return:
    """
    x = x.reshape(1,-1)
    #y_pred,sigma = gp_model.predict(x)
    y_pred, sigma = gp_model.predict(x)
    sigma = abs(sigma)

    if type==0:
        y_pred_constraint1, sigma_constraint1 = inference(x[0],nodes[0])  # prediction for the constraint function
    else:
        y_pred_constraint1, sigma_constraint1 = nodes[0].model.predict(x)

    #calculate the contribution of the constraint function
    sigma_constraint1 = abs(sigma_constraint1)
    Z_constraint1 = (0-y_pred_constraint1)/np.sqrt(sigma_constraint1)
    #print(y_pred_constraint1,norm.cdf(Z_constraint1))

    #Calculate normalized value
    Z = (-y_pred+min_val)/np.sqrt(sigma)
    #Calculate CDF and PDF for the given normal value
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    #calculate aquisition
    acq = np.sqrt(sigma)*(Z*Phi_Z+phi_Z)*((norm.cdf(Z_constraint1))**temperature)
    return (np.float(-acq))

def optimize_gaussian_process(model,model_constraint,nodes,x_array,type):
    """
    :This finds the best guess so-far
    :param model: Created model for the cost-function
    :param model_constraint: Created model for the constraint function
    :param nodes: List of created nodes (using piece-wise GP)
    :param x_array: list of all parameter values
    :param type: piece-wise or regular Gaussian process
    :return: a pair of best values (both parameter and value)
    """
    mean_fun, cov_fun = model.predict(x_array)  # find mean and covariance function for the loss function
    param_list = []
    loss_values = []
    for x in x_array: param_list.append(x)
    for y in mean_fun: loss_values.append(y)
    x_min = None
    y_min = +100000000000
    for index, y in enumerate(loss_values):
        # calculate constraint CDF
        p = param_list[index].reshape(1, -1)
        #y_pred_constraint1, std_constraint1 = model_constraint.predict(p)
        if type==0:
            y_pred_constraint1,std_constraint1 = inference(p[0],nodes[0])
        else:
            y_pred_constraint1, std_constraint1 = model_constraint.predict(p)
            #y_pred_constraint1, std_constraint1 = model_constraint.predict(p,return_cov=True)

        constraint_cdf = norm.cdf((0 - y_pred_constraint1) / (np.sqrt(std_constraint1) ))
        if y[0] < y_min and constraint_cdf > .95:
            y_min = y[0];
            x_min = param_list[index]

    return (x_min,y_min)

if __name__=="__main__":
    tree_error = []
    gp_error = []
    num_samples = 200
    N_D = 1
    MIN = []
    MAX = []
    writer = open("standard_deviation_experiment.txt","w")
    subplot_index = 0
    gp_vals = {}
    pc_vals = {}
    for power in [2]:
        gp_vals[power] = {}
        pc_vals[power] = {}
        gp_cost_map = {}
        pc_cost_map = {}
        for std in [.01,.1,.2,.3,.5,.7]:
            gp_cost_map[std] = []
            pc_cost_map[std] = []
            subplot_index+=1
            tree_error = []
            gp_error = []
            mcmmc=0
            while mcmmc<100:
                min_x = -5
                max_x = 5
                [MIN.append(min_x) for n in range(0,N_D)]
                [MAX.append(max_x) for n in range(0,N_D)]
                #create x_vals vector
                x_vals = []
                for n in range(0,num_samples):
                    x = (max_x-min_x)*random.random()+min_x
                    y = (max_x - min_x) * random.random() + min_x
                    if N_D>1:
                        x_vals.append([x,y])
                    else:
                        x_vals.append([x])

                x_array = []
                [x_array.append(x) for x in x_vals]
                x_array = np.array(x_array)

                training_x = []
                training_y = []
                training_val = []
                [training_x.append(z[0]) for z in x_array]
                if N_D>1:[training_y.append(z[1]) for z in x_array]

                y_array = []
                y_cost_array = []
                for x in x_array: y_array.append(sample_function(x,std)[1]) #real constraint values
                for x in x_array: y_cost_array.append(sample_cost_function(x,power)) #real cost values
                y_array = np.array(y_array)
                y_cost_array = np.array(y_cost_array)
                [training_val.append(z[0]) for z in y_array]

                #build model based on all the data
                model = GPy.models.GPRegression(x_array,y_array)
                model.optimize("bfgs")
                model_cost = GPy.models.GPRegression(x_array, y_cost_array)
                model_cost.optimize("bfgs")

                mean,var = model.predict(x_array)
                model_quality = np.mean(var)
                nodes = [node(model,x_array,y_array,MIN,MAX,model_quality,0,None,np.mean(model_quality),np.var(model_quality))]
                build_gp_tree(nodes,1)
                x_min_regular,y_min_regular = optimize_gaussian_process(model_cost, model, nodes, x_array, 1)
                x_min_pwise, y_min_pwise = optimize_gaussian_process(model_cost, model, nodes, x_array, 0)

                reg_error = []
                pc_error = []
                for x in x_array:
                    x = x.reshape(1, -1)
                    mean, var = nodes[0].model.predict(x)
                    mean_pc, var_pc = inference(x[0], nodes[0])
                    if x > 0:
                        reg_error.append(abs(1 - mean))
                        pc_error.append(abs(1-mean_pc))
                    else:
                        reg_error.append(abs(-1 - mean))
                        pc_error.append(abs(-1-mean_pc))

                reg_norm = np.linalg.norm(reg_error)
                pc_norm = np.linalg.norm(pc_error)
                ratio = reg_norm/pc_norm
                if ratio<10:
                    continue
                else:
                    mcmmc+=1

                print(mcmmc)

                min_x = -5
                max_x = 5
                N_D = 1
                condition = False
                temperature = 1
                prev_error = None
                no_change_cons = 0

                # optimization based on treed GP
                terminate = False
                step = 0
                positive = True
                while not terminate:
                    found = False
                    while not found:
                        try:
                            opt = nlopt.opt(nlopt.GN_DIRECT, N_D)
                            lb = np.array([np.float(min_x)])
                            ub = np.array([np.float(max_x)])
                            opt.set_lower_bounds(lb)
                            opt.set_upper_bounds(ub)
                            opt.set_maxeval(1000)
                            opt.set_min_objective(lambda x, grad: acquisition(x, grad, model_cost, nodes, y_min_regular, temperature, 1))
                            x0 = np.array([np.float(random.random())])
                            x = opt.optimize(x0)
                            found = True
                            terminate = True
                        except:
                            print("error")
                            pass
                x_opt_reg = x

                # optimization based on regular GP
                terminate = False
                while not terminate:
                    found = False
                    while not found:
                        try:
                            opt = nlopt.opt(nlopt.GN_DIRECT, N_D)
                            lb = np.array([np.float(min_x)])
                            ub = np.array([np.float(max_x)])
                            opt.set_lower_bounds(lb)
                            opt.set_upper_bounds(ub)
                            opt.set_maxeval(1000)
                            opt.set_min_objective(lambda x, grad: acquisition(x, grad, model_cost, nodes, y_min_pwise, temperature, 0))
                            x0 = np.array([np.float(random.random())])
                            x = opt.optimize(x0)
                            found = True
                            terminate = True
                        except:
                            print("error")
                            pass

                x_opt_pwise = x
                a = x_opt_reg ** power
                b = x_opt_pwise ** power

                gp_cost_map[std].append(a[0])
                pc_cost_map[std].append(b[0])

                print(gp_cost_map)
                print(pc_cost_map)
        gp_vals[power] = gp_cost_map
        pc_vals[power] = pc_cost_map
import joblib
import csv
import os
import copy
from config import *
from labels_config import *
from progress import *
from utils import *
from tqdm import tqdm
from model import Normalize,attack_Net, init_weights, mi_net, sfe_mi_net, ltfe_mi_net
from scipy.stats.mstats import rankdata
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from numpy import trapz
import torch.nn as nn
from sfe_utils import test_sfe,extract_features_sfe, labeldata_sfe
from ltfe_utils import test_ltfe, labeldata_ltfe
from ctfe_utils import test_ctfe, labeldata_ctfe
from SFELearning import load_extractor, load_classifier
from LTFELearning import load_classifier as load_classifier_ltfe, concatenate
from sklearn.metrics import accuracy_score, auc, roc_curve


def get_norms(norm_filename, device='cpu'):
    Ave, std = joblib.load(norm_filename)
    Ave = Ave.float().to(device)
    std = std.float().to(device)
    return Ave, std

def mse_loss(x, y):
    return (((x - y))**2).sum()/len(x)

def get_activations(model, x):
    model.eval()
    x = x.float()
    with torch.no_grad():
        f1,f2,f3,f4 = model.get_all_features(x)
        return [f1,f2,f3,f4]
        
def get_activations_ltfe(extractor, classifier, x, feature):
    classifier.eval()
    extractor.eval()
    x = x.float()
    with torch.no_grad():
        f1,f2,f3,_ = extractor.get_all_features(x)
        f4 = classifier(feature)
        return [f1,f2,f3,f4]

def get_gradients_ctfe(model, x, y):
    for i in range(len(x)):
        out = model(x[i])
        loss = mse_loss(out, y[i])
        loss.backward()
        if i == 0:
           fc1, fc2, fc3, fc4, loss_l = model.feature_ex.fc1.weight.grad.unsqueeze(0), model.feature_ex.fc2.weight.grad.unsqueeze(0), model.feature_ex.fc3.weight.grad.unsqueeze(0), model.classifier_m.fc_1.weight.grad.unsqueeze(0), loss.detach().unsqueeze(0)
        else:
           fc1 = torch.cat((fc1, model.feature_ex.fc1.weight.grad.unsqueeze(0)))
           fc2 = torch.cat((fc2, model.feature_ex.fc2.weight.grad.unsqueeze(0)))
           fc3 = torch.cat((fc3, model.feature_ex.fc3.weight.grad.unsqueeze(0)))
           fc4 = torch.cat((fc4, model.classifier_m.fc_1.weight.grad.unsqueeze(0)))
           loss_l = torch.cat((loss_l, loss.detach().unsqueeze(0)))
    return [fc1,fc2,fc3,fc4,loss_l,y.float()]

def get_gradients_sfe(model, x, y):
    for i in range(len(x)):
        out = model(x[i])
        loss = mse_loss(out, y[i])
        loss.backward()
        if i == 0:
           fc1, loss_l = model.fc_1.weight.grad.unsqueeze(0), loss.detach().unsqueeze(0)
        else:
           fc1 = torch.cat((fc1, model.fc_1.weight.grad.unsqueeze(0)))
           loss_l = torch.cat((loss_l, loss.detach().unsqueeze(0)))
    return [fc1,loss_l,y.float()]
    
def get_gradients_ltfe(extractor, classifier, x, y):
    for i in range(len(x)):
        features=  []
        for j in range(len(extractor)):
            extract_features = extractor[j].extractor(x[i:i+1])
            features.append(extract_features)
        features = torch.cat(features,1)
        out = classifier(features)
        loss = mse_loss(out, y[i])
        loss.backward()
        if i == 0:
           fc1, fc2, fc3, loss_l = extractor[0].feature_ex.fc1.weight.grad.unsqueeze(0), extractor[0].feature_ex.fc2.weight.grad.unsqueeze(0), extractor[0].feature_ex.fc3.weight.grad.unsqueeze(0), loss.detach().unsqueeze(0)
           fc4 = classifier.fc_1.weight.grad.unsqueeze(0)
        else:
           fc1 = torch.cat((fc1, extractor[0].feature_ex.fc1.weight.grad.unsqueeze(0)))
           fc2 = torch.cat((fc2, extractor[0].feature_ex.fc2.weight.grad.unsqueeze(0)))
           fc3 = torch.cat((fc3, extractor[0].feature_ex.fc3.weight.grad.unsqueeze(0)))
           fc4 = torch.cat((fc4, classifier.fc_1.weight.grad.unsqueeze(0)))
           loss_l = torch.cat((loss_l, loss.detach().unsqueeze(0)))
    return [fc1,fc2,fc3,fc4,loss_l,y.float()]

def generate_new_features_ltfe(extractors, classifier, x, y, args, norms):
    x = Normalize(norms[0], norms[1], x)
    features, _ = concatenate(x.float(), [], extractors, int(args.soc_attacker), args)
    result = get_activations_ltfe(extractors[0], classifier, x, features)
    result.extend(get_gradients_ltfe(extractors,classifier, x, y))
    return result

def generate_new_features_ctfe(model, x, y, norms):
    x = Normalize(norms[0], norms[1], x)
    result = get_activations(model, x)
    result.extend(get_gradients_ctfe(model, x, y))
    return result

def generate_new_features_sfe(extractor_model, classifier, x, y, norms):
    x = Normalize(norms[0], norms[1], x)
    with torch.no_grad():
         features = extractor_model.extractor(x)
    result = [classifier(features)]
    result.extend(get_gradients_sfe(classifier, features, y))
    return result

def train_attacker(x_member, y_member, x_non_member, y_non_member, extractor, classifier, args):
    norms = load_norms(args.method,args.dataset,args.device)
    pct_of_training = 0.1*5
    args.batch_size = 128*2

    indices = np.arange(len(x_member[0]))
    member_indices = np.random.choice(indices, int(pct_of_training*len(indices)), replace=False)
    np.save('./data/member_indices_'+args.dataset+'_'+args.method+'.npy',member_indices)

    indices = np.arange(len(x_non_member[0]))
    non_member_indices = np.random.choice(indices, int(pct_of_training*len(indices)), replace=False)
    np.save('./data/non_member_indices_'+args.dataset+'_'+args.method+'.npy',non_member_indices)

    x_member = x_member[0][member_indices]
    y_member = y_member[0][member_indices]
   
    x_non_member = x_non_member[0][non_member_indices]
    y_non_member = y_non_member[0][non_member_indices]
    
    x = torch.cat((x_member, x_non_member))
    z = torch.cat((y_member, y_non_member))

    y = torch.ones(len(x_member))
    y = torch.cat((y, torch.zeros(len(x_non_member))))

    if args.method == 'ctfe':
       model = mi_net(input_shape=args.n_features, output_shape = args.n_classes).to(args.device)
    elif args.method == 'sfe':
       model = sfe_mi_net(input_shape=args.n_features, output_shape = args.n_classes).to(args.device)
    else:
       model = ltfe_mi_net(input_shape=args.n_features, output_shape = args.n_classes).to(args.device)
#    model.apply(init_weights)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    n_datapoints = len(x)
    all_indices = np.arange(n_datapoints)
    batches_per_epoch = int(n_datapoints/args.batch_size)
    n_subset_indices = batches_per_epoch*args.batch_size
    criterion = nn.MSELoss()
    
    labels_elems = []
    labels_elems.append((0, 'member'))

    n_epochs = 200
    np.random.shuffle(all_indices)
    t = torch.Tensor([args.threshold]).to(args.device)
    best_acc = 0
    for i in range(n_epochs):
        running_loss = 0.0
        correct, tot = 0, 0
        acc = {}
        for l_e in labels_elems:
            acc[l_e[1]] = AccuracyMeasurer()
        for j in range(0, n_datapoints, args.batch_size):
            if j == n_subset_indices:
               batch_indices = all_indices[j:]
            else:
               batch_indices = all_indices[j:j+args.batch_size]
            inputs, targets, labels = x[batch_indices].float(), y[batch_indices], z[batch_indices][:,args.desired_labels]

            inputs, targets, labels = inputs.to(args.device), targets.to(args.device), labels.to(args.device)
            if args.method == 'sfe':
               yp = generate_new_features_sfe(extractor, classifier[0], inputs, labels, norms)
            elif args.method == 'ltfe':
               yp = generate_new_features_ltfe(extractor[0], classifier[0], inputs, labels, args, norms)
            elif args.method == 'ctfe':
               yp = generate_new_features_ctfe(classifier[0], inputs, labels, norms)
            outputs = model(yp)
            loss = criterion(outputs, targets.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            outputs = (outputs > t).float() * 1
            correct_in_batch, tot_in_batch = update_acc_for_batch(batch_indices, n_datapoints, outputs, targets.unsqueeze(1), acc, labels_elems)
            correct += correct_in_batch
            tot += tot_in_batch
            progress_bar(j, n_datapoints, 'Loss: %f | Acc: %.3f%% (%d/%d)' % (running_loss/(j+1), 100.*correct/tot, correct, tot))
#            scheduler.step()
        print('\nEpoch {} / {}, Training Accuracy {}% ({} / {})'.format(i+1, n_epochs, correct *100.0/tot, correct, tot))
        for l_e in labels_elems:
            print(l_e[1], " ", acc[l_e[1]].get_accuracy_precision_recall_F1_str(), "; actual_positives: ", acc[l_e[1]].get_actual_positives(), ";  total: ", acc[l_e[1]].get_total_processed())
        if (100.*correct/tot) >= best_acc:
           print("Saving!")
           torch.save(model.state_dict(), args.models_folder+'/attack_model_'+args.method+'.pth')
           best_acc = (100.*correct/tot)

def per_label_attack_performance(x_member, y_member, x_non_member, y_non_member, extractor, classifier, args, all=False):
    norms = load_norms(args.method,args.dataset,args.device)

    indices = np.arange(len(x_non_member[0]))
    non_member_indices = np.load('./data/non_member_indices_'+args.dataset+'_'+args.method+'.npy')
    non_member_indices = np.setdiff1d(list(range(len(indices))), non_member_indices)

    indices = np.arange(len(x_member[0]))
    print(len(non_member_indices))
    member_indices = np.random.choice(indices, len(non_member_indices), replace=False)
#    member_indices = np.load('./data/member_indices_'+args.dataset+'_'+args.method+'.npy')
#    member_indices = np.setdiff1d(list(range(len(indices))), member_indices)

    x_member = x_member[0][member_indices]
    y_member = y_member[0][member_indices]

    x_non_member = x_non_member[0][non_member_indices]
    y_non_member = y_non_member[0][non_member_indices]

#    if all:
 #      indices = np.arange(len(x_member))
  #     mindices = np.random.choice(indices, args.n_points, replace=False)
   #    indices = np.arange(len(x_non_member))
    #   nindices = np.random.choice(indices, args.n_points, replace=False)
    #else:
     #  indices = np.arange(len(x_member))
      # mindices = np.random.choice(indices, args.n_points, replace=False)
       #indices = np.arange(len(x_non_member))
       #nindices = np.random.choice(indices, args.n_points, replace=False)
    mindices = np.arange(len(x_member))
    nindices = np.arange(len(x_non_member))


    x = torch.cat((x_member[mindices], x_non_member[nindices]))
    z = torch.cat((y_member[mindices], y_non_member[nindices]))

    y = torch.ones(len(x_member[mindices]))
    y = torch.cat((y, torch.zeros(len(x_non_member[nindices]))))

    if args.method == 'ctfe':
       model = mi_net(input_shape=args.n_features, output_shape = args.n_classes)
    elif args.method == 'sfe':
       model = sfe_mi_net(input_shape=args.n_features, output_shape = args.n_classes)
    else:
       model = ltfe_mi_net(input_shape=args.n_features, output_shape = args.n_classes)
    model.load_state_dict(torch.load(args.models_folder+'/attack_model_'+args.method+'.pth', map_location=args.device))
    model = model.to(args.device)
    model.eval()
    labels_elems = []
    labels_elems.append((0, 'member'))

    t = torch.Tensor([0.5]).to(args.device)

    if all:
       desired_labels = [0]
       args.batch_size = 5000
    else:
       desired_labels = args.desired_labels
       args.batch_size = 1280
    for label in desired_labels:
        if not all:
           input   = x[z[:,label]==1]
           target  = y[z[:,label]==1]
           labelss = z[z[:,label]==1]
        else:
           input  = x
           target = y
           labelss  = z       
        labelss = labelss[:,args.desired_labels]
        n_datapoints = len(input)
        all_indices = np.arange(n_datapoints)
        batches_per_epoch = int(n_datapoints/args.batch_size)
        n_subset_indices = batches_per_epoch*args.batch_size
        np.random.shuffle(all_indices)
        acc = {}
        for l_e in labels_elems:
            acc[l_e[1]] = AccuracyMeasurer()
        for j in tqdm(range(0, n_datapoints, args.batch_size)):
            if j == n_subset_indices:
               batch_indices = all_indices[j:]
            else:
               batch_indices = all_indices[j:j+args.batch_size]    
            inputs, labels, targets = input[batch_indices].float().to(args.device), labelss[batch_indices].to(args.device), target[batch_indices].to(args.device)
                
            if args.method == 'sfe':
               yp = generate_new_features_sfe(extractor, classifier[0], inputs, labels, norms)
            elif args.method == 'ltfe':
               yp = generate_new_features_ltfe(extractor[0], classifier[0], inputs, labels, args, norms)
            elif args.method == 'ctfe':
               yp = generate_new_features_ctfe(classifier[0], inputs, labels, norms)
            with torch.no_grad():
                 outs = model(yp)
            outs = (outs >= t).float() * 1
            update_acc_for_batch(batch_indices, n_datapoints, outs, targets.unsqueeze(1), acc, labels_elems)
        if not all:
           print(args.labels_names[label])
        for l_e in labels_elems:
            print(l_e[1], " ", acc[l_e[1]].get_accuracy_precision_recall_F1_str(), "; actual_positives: ", acc[l_e[1]].get_actual_positives(), ";  total: ", acc[l_e[1]].get_total_processed())
        
def plot_roc(x_member, y_member, x_non_member, y_non_member, extractor, classifier, args):
    norms = load_norms(args.method,args.dataset,args.device)

    indices = np.arange(len(x_non_member[0]))
    non_member_indices = np.load('./data/non_member_indices_'+args.dataset+'_'+args.method+'.npy')
    non_member_indices = np.setdiff1d(list(range(len(indices))), non_member_indices)

    indices = np.arange(len(x_member[0]))
    member_indices = np.random.choice(indices, len(non_member_indices), replace=False)

    x_member = x_member[0][member_indices]
    y_member = y_member[0][member_indices]
   
    x_non_member = x_non_member[0][non_member_indices]
    y_non_member = y_non_member[0][non_member_indices]
    
    args.batch_size = 1280
    indices = np.arange(len(x_member))
    mindices = np.random.choice(indices, args.n_points, replace=False)
    indices = np.arange(len(x_non_member))
    nindices = np.random.choice(indices, args.n_points, replace=False)
    
    x = torch.cat((x_member[mindices], x_non_member[nindices]))
    z = torch.cat((y_member[mindices], y_non_member[nindices]))

    y = torch.ones(len(x_member[mindices]))
    y = torch.cat((y, torch.zeros(len(x_non_member[nindices]))))
    
    if args.method == 'ctfe':
       model = mi_net(input_shape=args.n_features, output_shape = args.n_classes)
    elif args.method == 'sfe':
       model = sfe_mi_net(input_shape=args.n_features, output_shape = args.n_classes)
    else:
       model = ltfe_mi_net(input_shape=args.n_features, output_shape = args.n_classes)
    model.load_state_dict(torch.load(args.models_folder+'/attack_model_'+args.method+'.pth', map_location=args.device))
    model = model.to(args.device)
    model.eval()
    
    n_datapoints = len(x)
    all_indices = np.arange(n_datapoints)
    batches_per_epoch = int(n_datapoints/args.batch_size)
    n_subset_indices = batches_per_epoch*args.batch_size

    labels_elems = []
    labels_elems.append((0, 'member'))
    np.random.shuffle(all_indices)
    
    tpr = []
    fpr = []
    thresholds = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for thr in thresholds:
        t = torch.Tensor([thr]).to(args.device)
        acc = {}
        for l_e in labels_elems:
            acc[l_e[1]] = AccuracyMeasurer()
        
        for j in tqdm(range(0, n_datapoints, args.batch_size)):
            if j == n_subset_indices:
               batch_indices = all_indices[j:]
            else:
               batch_indices = all_indices[j:j+args.batch_size]
            inputs, labels = x[batch_indices].float().to(args.device), z[batch_indices][:,args.desired_labels].to(args.device)
            targets = y[batch_indices].to(args.device)
            if args.method == 'sfe':
               yp = generate_new_features_sfe(extractor, classifier[0], inputs, labels, norms)
            elif args.method == 'ltfe':
               yp = generate_new_features_ltfe(extractor[0], classifier[0], inputs, labels, args, norms)
            elif args.method == 'ctfe':
               yp = generate_new_features_ctfe(classifier[0], inputs, labels, norms)
            outs = model(yp)
            outs = (outs >= t).float() * 1
            update_acc_for_batch(batch_indices, n_datapoints, outs, targets.unsqueeze(1), acc, labels_elems)
        for l_e in labels_elems:
            tpr.append(acc[l_e[1]]. get_tpr_fpr()[0])
            fpr.append(acc[l_e[1]]. get_tpr_fpr()[1])


    fig = plt.figure(2)
    np.save('data/fpr_'+args.method+'.npy',fpr)
    np.save('data/tpr_'+args.method+'.npy',tpr)
    plt.title('ROC of Membership Inference Attack '+args.method)
    roc_auc = trapz(tpr[::-1], fpr[::-1])
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('Figs/roc_'+args.method+'.png')
    plt.close()


def plot_roc_all():
    fig = plt.figure(2)
    fpr_sfe = np.load('data/fpr_sfe.npy')
    tpr_sfe = np.load('data/tpr_sfe.npy')
    roc_auc_sfe = trapz(tpr_sfe[::-1], fpr_sfe[::-1])
    plt.title('ROC of Membership Inference Attack')
    plt.plot(fpr_sfe, tpr_sfe, 'b', label='SFE AUC = %0.2f' % roc_auc_sfe)

    fpr_ctfe = np.load('data/fpr_ctfe.npy')
    tpr_ctfe = np.load('data/tpr_ctfe.npy')
    roc_auc_ctfe = trapz(tpr_ctfe[::-1], fpr_ctfe[::-1])
    plt.plot(fpr_ctfe, tpr_ctfe, 'g', label='CTFE AUC = %0.2f' % roc_auc_ctfe)

    fpr_ltfe = np.load('data/fpr_ltfe.npy')
    tpr_ltfe = np.load('data/tpr_ltfe.npy')
    roc_auc_ltfe = trapz(tpr_ltfe[::-1], fpr_ltfe[::-1])
    plt.plot(fpr_ltfe, tpr_ltfe, 'c', label='LTFE AUC = %0.2f' % roc_auc_ltfe)

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('Figs/roc.png')
    plt.close()

def plot_privacy_risk(x_member, y_member, x_non_member, y_non_member, extractor, classifier, args):
    norms = load_norms(args.method,args.dataset,args.device)

    indices = np.arange(len(x_non_member[0]))
    non_member_indices = np.load('./data/non_member_indices_'+args.dataset+'_'+args.method+'.npy')
    non_member_indices = np.setdiff1d(list(range(len(indices))), non_member_indices)

    indices = np.arange(len(x_member[0]))
    member_indices = np.random.choice(indices, len(non_member_indices), replace=False)


    x_member = x_member[0][member_indices]
    y_member = y_member[0][member_indices]
   
    x_non_member = x_non_member[0][non_member_indices]
    y_non_member = y_non_member[0][non_member_indices]
    '''
    args.batch_size = 1280
    indices = np.arange(len(x_member))
    mindices = np.random.choice(indices, args.n_points, replace=False)
    indices = np.arange(len(x_non_member))
    nindices = np.random.choice(indices, args.n_points, replace=False)
    '''
    mindices = np.arange(len(x_member))
    nindices = np.arange(len(x_non_member))
    x = torch.cat((x_member[mindices], x_non_member[nindices]))
    z = torch.cat((y_member[mindices], y_non_member[nindices]))

    y = torch.ones(len(x_member[mindices]))
    y = torch.cat((y, torch.zeros(len(x_non_member[nindices]))))
    
    if args.method == 'ctfe':
       model = mi_net(input_shape=args.n_features, output_shape = args.n_classes)
    elif args.method == 'sfe':
       model = sfe_mi_net(input_shape=args.n_features, output_shape = args.n_classes)
    else:
       model = ltfe_mi_net(input_shape=args.n_features, output_shape = args.n_classes)
    model.load_state_dict(torch.load(args.models_folder+'/attack_model_'+args.method+'.pth', map_location=args.device))
    model = model.to(args.device)
    model.eval()
    
    n_datapoints = len(x)
    all_indices = np.arange(n_datapoints)
    batches_per_epoch = int(n_datapoints/args.batch_size)
    n_subset_indices = batches_per_epoch*args.batch_size

    np.random.shuffle(all_indices)
    
    member_preds = []
    non_member_preds = []
        
    for j in tqdm(range(0, n_datapoints, args.batch_size)):
        if j == n_subset_indices:
           batch_indices = all_indices[j:]
        else:
           batch_indices = all_indices[j:j+args.batch_size]
        inputs, labels = x[batch_indices].float().to(args.device), z[batch_indices][:,args.desired_labels].to(args.device)
        targets = y[batch_indices].to(args.device)
        if args.method == 'sfe':
           yp = generate_new_features_sfe(extractor, classifier[0], inputs, labels, norms)
        elif args.method == 'ltfe':
           yp = generate_new_features_ltfe(extractor[0], classifier[0], inputs, labels, args, norms)
        elif args.method == 'ctfe':
           yp = generate_new_features_ctfe(classifier[0], inputs, labels, norms)
        with torch.no_grad():
             outs = model(yp)        
        member_preds.extend(outs[targets==1].cpu())
        non_member_preds.extend(outs[targets==0].cpu())
    fig = plt.figure(1)
    plt.hist(np.array(member_preds).flatten(), color='xkcd:blue', alpha=0.7, bins=20,
             histtype='bar', range=(0, 1), weights=(np.ones_like(member_preds) / len(member_preds)), label='Training Data (Members)')
    plt.hist(np.array(non_member_preds).flatten(), color='xkcd:light blue', alpha=0.7, bins=20,
             histtype='bar', range=(0, 1), weights=(np.ones_like(non_member_preds) / len(non_member_preds)), label='Population Data (Non-members)')
    plt.xlabel('Membership Probability')
    plt.ylabel('Fraction')
    plt.title('Privacy Risk '+args.method)
    plt.legend(loc='upper left')
    plt.savefig('Figs/privacy_risk_'+args.method+'.png')
    plt.close()

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from model import FitnessPredictor
from data import FitnessDataset

def run_infer(batch_size, data_path, protein, epoch, lambda_l2, state_dict_file):
    df = pd.read_csv('{}'.format(data_path))
    raw_seq_arr_test = df.loc[:,'raw_seq'].values
    mut_seq_arr_test = df.loc[:,'mut_seq'].values
    label_test = df.loc[:,'label'].values
    
    dataset = FitnessDataset(raw_seq_arr_test,mut_seq_arr_test,label_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_layers = 1 # number of Transformer encoder layers added after ESM model
    num_heads = 8
    block_embed_dim = 128
    model = FitnessPredictor(num_layers,num_heads,block_embed_dim)
    
    state_dict = torch.load(state_dict_file,map_location=device)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    output_list = []
    with torch.no_grad():
        for i, (batch_raw_seq,batch_mut_seq,batch_mask_raw,batch_mask_mut,batch_label,batch_raw_mutations,batch_mut_mutations) in enumerate(loader):
            batch_raw_seq,batch_mut_seq,batch_mask_raw,batch_mask_mut,batch_label = batch_raw_seq.to(device),batch_mut_seq.to(device),batch_mask_raw.to(device),batch_mask_mut.to(device),batch_label.to(device)
            batch_raw_mutations, batch_mut_mutations = batch_raw_mutations.to(device), batch_mut_mutations.to(device)

            output, _, _ = model(batch_raw_seq,batch_mut_seq,batch_mask_raw,batch_mask_mut,batch_raw_mutations,batch_mut_mutations)
            
            output = output[:,1].cpu().detach().numpy()
            batch_label = batch_label.cpu().numpy()
            output_list.extend(output)
            
    return output_list

if __name__ == '__main__':
    path = '/home/datasets/cschwang/escape/evaluation/escape/DMS_subset_evescape/DMS_subset/'
    protein = 'Nipah_RBP'
    epoch = 1
    lambda_l2 = 0.05
    state_dict_file = './model_epoch{}_{}_lambdal2{}.pt'.format(epoch,protein,lambda_l2)
    
    batch_size = 1
    
    df_res = pd.DataFrame()
        
    data_path = f'{path}test_{protein}_gamma_pairs_no_aug_independent.csv'
        
    pred_list = run_infer(batch_size, data_path, protein, epoch, lambda_l2, state_dict_file)
    df_res['KEScape_score'] = pred_list
    df_res.to_csv('./KEScape_score_{}.csv'.format(protein),index=False)        
        

    
    
        
    
    
    



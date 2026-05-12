import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from model import FitnessPredictor
from data import FitnessDataset
import argparse

def run_infer(batch_size, data_path, state_dict_file):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()
    
    state_dict_file = args.checkpoint_path
    data_path = args.data_path
    output_path = args.output_path
    batch_size = args.batch_size
    
    df_res = pd.DataFrame()
        
    pred_list = run_infer(batch_size, data_path, state_dict_file)
    df_res['KEScape_score'] = pred_list
    df_res.to_csv(output_path,index=False)        
        

    
    
        
    
    
    



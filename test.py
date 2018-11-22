import os
import torch

from model import DenseNet, octDataset
from train import test_epoch


def test(model, data_path, label_file, save, batch_size):
    
    data_list = os.listdir(data_path)
    dataset = octDataset(data_path, data_list, label_file, argument=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=0, collate_fn=dataset.collate_fn)
    
    if torch.cuda.is_available():
        model = model.cuda()

    model.load_state_dict(torch.load(os.path.join(save, 'v1_test_model.dat')))
    
    
    _, loss, error = test_epoch(model, dataloader, is_test=True)
    
    return error.avg


if __name__ == "__main__":
    path = "/home/youkun/ext_data/nostream/"
    label_file = "/home/youkun/ext_data/ns_label.csv"
    save_pth = "/home/youkun/OCT_Classification/results/"
    net = DenseNet(small_inputs=False)

    error = test(net, path, label_file, save_pth, 1)
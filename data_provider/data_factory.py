from data_provider.data_loader import Dataset_GBPCAD_hour, Dataset_NUMSOLD_day
from torch.utils.data import DataLoader

data_dict = {
    'gbpcad' : Dataset_GBPCAD_hour,
    'numsold' : Dataset_NUMSOLD_day
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    drop_last = True
    batch_size = args.batch_size
    freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        num_lags=args.num_lags,
        target=args.target,
        freq=freq
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader

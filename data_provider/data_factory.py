from data_provider.data_loader import Dataset_GBPCAD_hour, Dataset_NUMSOLD_day, Dataset_ETHUSD_hour, Dataset_WEATHER_day, Dataset_HOUSE_day, Dataset_US500USD_hour, Dataset_ELECTR, Dataset_AAPL
from torch.utils.data import DataLoader

data_dict = {
    'gbpcad' : Dataset_GBPCAD_hour,
    'numsold' : Dataset_NUMSOLD_day,
    'ethusd' : Dataset_ETHUSD_hour,
    'weather' : Dataset_WEATHER_day,
    'house_sales' : Dataset_HOUSE_day,
    'us500usd' : Dataset_US500USD_hour,
    'electricity' : Dataset_ELECTR,
    'aapl' : Dataset_AAPL,
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    drop_last = True
    batch_size = args.batch_size
    freq = args.freq

    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        num_lags=args.num_lags,
        num_entries=args.num_entries,
        target=args.target,
        freq=freq
    )
    if args.model == "ResNet":
        data_set.split_tensors()
    elif args.model == "CNN":
        data_set.to_cnn()
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader

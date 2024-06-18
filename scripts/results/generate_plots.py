import pandas as pd
import matplotlib.pyplot as plt
import os

columns_to_plot = ['TestCG', 'TrainCG', 'ValiCG']

def plot_column_accuracy(df, column_name, root):
    if column_name not in df.columns:
        print(f"Skipping {column_name}: Required column is missing in {root}.")
        return

    df[f'{column_name}_Accuracy'] = df[column_name].apply(lambda x: x * 100) # Accuracy on last predicted value

    plt.figure(figsize=(10, 5))
    plt.plot(df["Epoch"], df[f'{column_name}_Accuracy'], marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
#    plt.title(f'{column_name} Accuracy Over Epochs for {os.path.basename(root)}')
    plt.grid(True)

    # Save the plot in the same directory as the CSV file
    plot_filename = f"{column_name}_results_plot.png"
    plot_filepath = os.path.join(root, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

def plot_cg_from_csv(base_directory):
    for root, _, files in os.walk(base_directory):
        for filename in files:
            if filename == "results.csv":
                filepath = os.path.join(root, filename)
                try:
                    df = pd.read_csv(filepath)

                    # Check if necessary columns exist
                    if 'Epoch' not in df.columns:
                        print(f"Skipping {filepath}: 'Epoch' column is missing.")
                        continue
                    df = df.head(50)

                    for column in columns_to_plot:
                        plot_column_accuracy(df, column, root)

                except pd.errors.EmptyDataError:
                    print(f"Skipping {filepath}: Empty data error.")
                except Exception as e:
                    print(f"Skipping {filepath} due to an unexpected error: {e}")

base_directory = os.path.join(os.getcwd(), ".")
plot_cg_from_csv(base_directory)

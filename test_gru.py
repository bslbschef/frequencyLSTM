import matplotlib.pyplot as plt
from model import WindSpeedGRU
from util import *
from parameter import *
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt


sys.stdout = Logger(filename=save_pth_path + 'test_default.log', stream=sys.stdout)
loaded_model = WindSpeedGRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, flag_bidirectional=flag_bidirectional, time_flag=time_flag)
# loaded_model.load_state_dict(torch.load(save_pth_path + 'epoch_u_0200.pth')['model_state_dict'])
# loaded_model.load_state_dict(torch.load(save_pth_path + 'epoch_u_0730.pth')['model_state_dict'])
# loaded_model.load_state_dict(torch.load(save_pth_path + 'epoch_u_0540.pth')['model_state_dict'])
# loaded_model.load_state_dict(torch.load(save_pth_path + 'epoch_u_0470.pth')['model_state_dict'])
# loaded_model.load_state_dict(torch.load(save_pth_path + 'epoch_u_1410.pth')['model_state_dict'])
# loaded_model.load_state_dict(torch.load(save_pth_path + 'epoch_u_1040.pth')['model_state_dict'])
# loaded_model.load_state_dict(torch.load(save_pth_path + 'epoch_u_4000.pth')['model_state_dict'])
# loaded_model.load_state_dict(torch.load(save_pth_path + 'epoch_u_4820.pth')['model_state_dict'])
# loaded_model.load_state_dict(torch.load(save_pth_path + 'epoch_u_6000.pth')['model_state_dict'])
# loaded_model.load_state_dict(torch.load(save_pth_path + 'epoch_u_12580.pth')['model_state_dict'])
loaded_model.load_state_dict(torch.load(save_pth_path + 'epoch_u_14570.pth')['model_state_dict'])
# loaded_model.load_state_dict(torch.load(save_pth_path + 'epoch_u_19290.pth')['model_state_dict'])
loaded_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)

r2_scores = []
rmse_scores = []
count = 0
for i in test_csv_files:
    count += 1
    test_data = load_data(i).to(device)
    with torch.no_grad():
        inputs, targets = test_data[:, :12, :], test_data[:, 12:, :]
        temporary_inputs, temporary_targets = vector_transfer_targetFinal_noBatch2(window_size, inputs, targets)
        temporary_targets = temporary_targets[:, 0:1]
        predicted_wind = loaded_model(temporary_inputs).squeeze(0)
    # raw_labels = test_data.cpu().numpy().squeeze(-1)
    predicted_wind = predicted_wind.cpu().numpy()
    temporary_targets = temporary_targets.cpu().numpy().squeeze(-1)
    save_true, save_predicted = temporary_targets, predicted_wind
    # if type_name == '_u':
    #     save_true = true_labels[window_size-1:, 0]
    #     save_raw = raw_labels[window_size-1:, 0]
    # elif type_name == '_v':
    #     save_true = true_labels[window_size-1:, 1]
    #     save_raw = raw_labels[window_size-1:, 1]
    # elif type_name == '_w':
    #     save_true = true_labels[window_size-1:, 2]
    #     save_raw = raw_labels[window_size-1:, 2]
    # save_predicted = predicted_labels
    # size_time = len(save_predicted)
    # time_x = np.arange(size_time) / (20*60)
    #
    # save_true = save_true[1500:]
    # save_predicted = save_predicted[1500:]

    r2 = r2_score(save_true, save_predicted)
    rmse = np.sqrt(mean_squared_error(save_true, save_predicted))
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    #
    matrix_uv = np.vstack((save_true, np.squeeze(save_predicted)))
    matrix_uv = np.transpose(matrix_uv)
    header = "save_true, save_predicted"
    np.savetxt(save_pth_path + '/test/train/output' + str(count).zfill(2) + '_test.txt', matrix_uv, delimiter=',', header=header)

    plt.figure(figsize=(10, 5))
    plt.plot(save_true, label='Met')
    plt.plot(save_predicted, label='Predicted-UAV')
    plt.xlabel('Time (1/20 s)')
    plt.ylabel('Wind Speed (m/s)')
    plt.legend()
    plt.title('Test Set (run' + str(i) + ')')
    plt.show(block=False)
    # plt.show()
    # plt.savefig(save_pth_path + f'/test/Test_num_{str(i)}.png')

print('Test Set')
print('----------------------------------------')
print(r2_scores)
print(rmse_scores)
print(f"Mean R2 Score: {np.mean(r2_scores):.4f}")
print(f"Mean RMSE Score: {np.mean(rmse_scores):.4f}")
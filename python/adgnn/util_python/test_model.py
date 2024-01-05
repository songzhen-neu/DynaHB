import adgnn.context.context as context
import torch
def test_model(model, test_dataset):
    cost_test = 0
    model.eval()
    model.to('cpu')
    hidden_state = [None for i in range(len(context.glContext.config['hidden']))]
    for time, snapshot in enumerate(test_dataset):
        y_hat, hidden_state = model(snapshot.x, snapshot.edge, snapshot.edge_weight, hidden_state,
                                    snapshot.deg)
        cost_test = cost_test + torch.mean((y_hat.view(-1) - snapshot.y) ** 2)
    cost_test = cost_test / (time + 1)
    model.to(context.glContext.config['device'])
    test_num = test_dataset.target_vertex[0][0].shape[0]
    return test_num, cost_test
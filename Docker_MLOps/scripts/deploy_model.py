import pickle

def deploy_model(model_file, deploy_path):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    with open(deploy_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    deploy_model('models/decision_tree_model.pkl', 'models/deployed_model.pkl')
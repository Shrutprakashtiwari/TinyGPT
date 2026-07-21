torch.save(model.state_dict(), "tinygpt.pth")
from google.colab import files
files.download("tinygpt.pth")

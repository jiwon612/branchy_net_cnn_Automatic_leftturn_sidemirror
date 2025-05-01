# branchy_net_cnn_Automatic_leftturn_sidemirror
![image](https://github.com/user-attachments/assets/22c6d754-f6b8-4601-9581-cd5dcf289a70)


## background


Inspired by the automatic low-down side mirror, it was created to eliminate blind spots when entering or merging intersections.
Internally, rather than using a general neural network(CNN, RNN, etc.), we use BranchyNet(Early Exit Network) to enable faster decisions in situations where road conditions are clear or more computation is not necessarily required.
### Sometimes faster computations can help our safety more than higher accuracy.
![image](https://github.com/user-attachments/assets/7ffd7b5e-e256-450e-aa7a-38446d1b925d)

![image](https://github.com/user-attachments/assets/ba6bbe8e-cf58-4c3a-82f1-7e8dc99f7336)


## Setup

Python 3.8 (for `dataclass` support) or higher is required.

pip install -r requirements.txt

or

```
pip install h5py
pip install keras
```


## Result
### Ours
![image](https://github.com/user-attachments/assets/8897c1b2-1948-466a-a6e9-dbf9e58c5aef)
![image](https://github.com/user-attachments/assets/fa983718-1c1c-4dd1-beef-3621bbe01061)


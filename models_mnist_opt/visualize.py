# visualize.py
# A complete script to load your trained CSNN and create an interactive web demo.


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import transforms
from snntorch import spikeplot as splt
from snntorch import surrogate
import snntorch as snn
from PIL import Image

# --- Imports for Streamlit ---
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# --- (Model classes and loading function remain the same) ---
class MemoryOptimizedCSNN(nn.Module):
    def __init__(self, beta=0.9, slope=25, num_classes=10, dropout_rate=0.1):
        super().__init__()
        self.spike_grad = surrogate.fast_sigmoid(slope=slope); self.beta = beta; self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(32); self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad); self.dropout1 = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(64); self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad); self.dropout2 = nn.Dropout2d(dropout_rate)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False); self.bn3 = nn.BatchNorm2d(128); self.lif3 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad); self.dropout3 = nn.Dropout2d(dropout_rate)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False); self.bn4 = nn.BatchNorm2d(256); self.lif4 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)); self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 128, bias=False); self.bn_fc1 = nn.BatchNorm1d(128); self.lif5 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad); self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 10, bias=False); self.lif6 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
    def forward(self, x, num_steps): pass

class CSNN_Visual(MemoryOptimizedCSNN):
    def forward(self, x, num_steps):
        mem1=self.lif1.init_leaky(); mem2=self.lif2.init_leaky(); mem3=self.lif3.init_leaky(); mem4=self.lif4.init_leaky(); mem5=self.lif5.init_leaky(); mem6=self.lif6.init_leaky()
        recordings={"spk1":[], "spk5":[], "spk6":[]}; x_norm=torch.clamp(x, 0, 1)
        for step in range(num_steps):
            x_spikes=(torch.rand_like(x_norm)<x_norm).float(); cur1=self.bn1(self.conv1(x_spikes)); spk1,mem1=self.lif1(cur1,mem1); spk1_dropped=self.dropout1(spk1)
            cur2=self.bn2(self.conv2(spk1_dropped)); spk2,mem2=self.lif2(cur2,mem2); spk2_dropped=self.dropout2(spk2); cur3=self.bn3(self.conv3(spk2_dropped)); spk3,mem3=self.lif3(cur3,mem3); spk3_dropped=self.dropout3(spk3)
            cur4=self.bn4(self.conv4(spk3_dropped)); spk4,mem4=self.lif4(cur4,mem4); pooled=self.global_avg_pool(spk4); flat=self.flatten(pooled); cur5=self.bn_fc1(self.fc1(flat)); spk5,mem5=self.lif5(cur5,mem5); spk5_dropped=self.dropout_fc(spk5)
            cur6=self.fc2(spk5_dropped); spk6,mem6=self.lif6(cur6,mem6); recordings["spk1"].append(spk1); recordings["spk5"].append(spk5); recordings["spk6"].append(spk6)
        for key, value in recordings.items(): recordings[key]=torch.stack(value, dim=0)
        return recordings

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_vis = CSNN_Visual().to(device)
    MODEL_FILENAME = 'best_model_for_42_memory_optimized.pth'
    try:
        checkpoint = torch.load(MODEL_FILENAME, map_location=device)
        model_vis.load_state_dict(checkpoint['model_state_dict'])
        model_vis.eval()
        return model_vis, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model_vis, device = load_model()

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Resize((28, 28), antialias=True), transforms.Normalize((0,), (1,))
])

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="SNN Visualization" ,page_icon="🧠")

st.title(" Visualizing a Convolutional Spiking Neural Network (CSNN) ")
st.markdown("""
Welcome! This app lets you look inside a **Convolutional Spiking Neural Network (CSNN)**, a type of AI inspired by the brain's energy efficiency.
Draw a digit on the canvas in the sidebar to see how the network processes it over time using discrete **spikes**.
""")

# --- Sidebar for controls ---
with st.sidebar:
    st.header("Controls")
    st.markdown("Draw a digit (0-9)")
    canvas_result = st_canvas(
        stroke_width=20, stroke_color="white", background_color="black",
        height=280, width=280, drawing_mode="freedraw", key="canvas",
    )

# --- Main Area for Visualizations ---
if canvas_result.image_data is not None and model_vis is not None:
    
    pil_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    num_steps_demo = 30
    
    with torch.no_grad():
        recordings = model_vis(img_tensor, num_steps=num_steps_demo)
    
    prediction = torch.sum(recordings["spk6"], dim=0).argmax().item()
    for key in recordings: recordings[key] = recordings[key].detach().cpu().squeeze(1)

    # --- Display Results and Explanations ---
    st.header(f"Analysis of Digit: `{prediction}`")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.style.use('dark_background') 

    # Plot 1: Input
    axes[0, 0].imshow(img_tensor.cpu().squeeze().numpy(), cmap='gray'); axes[0, 0].set_title("1. Input to Model (28x28)"); axes[0, 0].axis('off')
    
    # Plot 2: Conv1 Spikes
    mid_step = num_steps_demo // 2; conv1_spk_map = recordings["spk1"][mid_step, 5, :, :]; axes[0, 1].imshow(conv1_spk_map, cmap='binary', interpolation='nearest'); axes[0, 1].set_title(f"2. Conv1 Spikes (T={mid_step})"); axes[0, 1].axis('off')
    
    # Plot 3: FC Layer Raster
    splt.raster(recordings["spk5"], axes[1, 0], s=2, c='white'); axes[1, 0].set_title("3. FC Layer Spike Raster"); axes[1, 0].set_xlabel("Time Step"); axes[1, 0].set_ylabel("Neuron Index"); axes[1, 0].set_xlim(0, num_steps_demo)

    # Plot 4: Output Spike Count
    spike_counts = torch.sum(recordings["spk6"], dim=0); axes[1, 1].bar(range(10), spike_counts, color='cyan'); axes[1, 1].set_title("4. Output Layer Spike Count (Vote)"); axes[1, 1].set_xlabel("Output Neuron (Digit)"); axes[1, 1].set_ylabel("Total Spikes"); axes[1, 1].set_xticks(range(10))
    
    plt.tight_layout(); st.pyplot(fig)

    # --- Explanations  ---
    with st.expander("What do these graphs mean? (Click to learn)"):
        st.markdown("""
        The four graphs above show the **flow of information** through the SNN:
        
        **1. Input to Model:** This is the processed 28x28 grayscale image that the network 'sees'.
        
        **2. Conv1 Spikes:** This is a snapshot of the activity in the first convolutional layer at the middle timestep. White pixels represent neurons firing spikes after detecting basic features like edges and curves.
        
        **3. FC Layer Spike Raster:** This graph shows the complete 'thought process' over 30 timesteps. Each dot is a spike from a neuron in a deeper layer, forming a unique temporal pattern for the digit you drew. This is the core of event-driven processing.
        
        **4. Output Layer Spike Count (Vote):** This is the final decision. The bar chart shows the total spikes fired by each of the 10 output neurons. The neuron with the most spikes determines the model's prediction.
        """)
else:
    st.info("The visualization will appear here once you draw a digit in the sidebar.")
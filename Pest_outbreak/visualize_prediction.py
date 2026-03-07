# Pest_outbreak/visualize_prediction.py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def visualize_prediction(image_path, predicted_class, confidence):
    """
    Visualize the prediction with the image
    """
    # Read and display image
    img = mpimg.imread(image_path)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    
    # Add prediction text
    title = f"Predicted: {predicted_class}\nConfidence: {confidence:.2%}"
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add color based on confidence
    if confidence > 0.8:
        color = 'green'
    elif confidence > 0.5:
        color = 'orange'
    else:
        color = 'red'
    
    plt.gca().add_patch(plt.Rectangle(
        (10, 10), 200, 60, 
        facecolor=color, alpha=0.7, edgecolor='white', linewidth=2
    ))
    
    plt.text(20, 45, f"Confidence: {confidence:.1%}", 
             color='white', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()

def save_prediction_viz(image_path, predicted_class, confidence, save_path):
    """
    Save the visualization to a file
    """
    fig = visualize_prediction(image_path, predicted_class, confidence)
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
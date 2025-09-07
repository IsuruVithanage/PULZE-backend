import base64
import io
from typing import List

from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
from xhtml2pdf import pisa
import os

# Set up Jinja2 to find templates in the 'templates' folder
# This path is relative to the root of your project where you run uvicorn
template_loader = FileSystemLoader(searchpath="./app/templates")
template_env = Environment(loader=template_loader)


def create_pdf_from_html(template_context: dict, output_path: str):
    """
    Renders an HTML template with the given context and converts it to a PDF.

    :param template_context: A dictionary of data to pass to the template.
    :param output_path: The file path where the generated PDF will be saved.
    """
    try:
        # Load the HTML template
        template = template_env.get_template("report_template.html")

        # Render the template with your data
        source_html = template.render(template_context)

        # Create the PDF file
        with open(output_path, "w+b") as result_file:
            pisa_status = pisa.CreatePDF(
                source_html,  # The HTML to convert
                dest=result_file  # File handle to receive result
            )

        # Check for errors
        if pisa_status.err:
            raise IOError(f"PDF creation error: {pisa_status.err}")

        print(f"Successfully created PDF at {output_path}")
        return output_path

    except Exception as e:
        print(f"An error occurred during PDF generation: {e}")
        # Re-raise the exception to be handled by the endpoint
        raise

def generate_trend_chart_base64(labels: List[str], data: List[float], metric_name: str, unit: str) -> str:
    """
    Generates a professional-looking trend chart and returns it as a Base64 encoded string.
    """
    plt.style.use('dark_background') # Use a style that fits a professional report
    fig, ax = plt.subplots(figsize=(8, 3.5)) # Create a figure and an axes

    # Plot the data
    ax.plot(labels, data, marker='o', linestyle='-', color='#21DB9A')

    # --- Styling ---
    # Set titles and labels
    ax.set_title(f'{metric_name} Trend', fontsize=14, color='white', pad=20)
    ax.set_ylabel(unit, fontsize=10, color='gray')
    ax.tick_params(axis='x', colors='gray', rotation=30)
    ax.tick_params(axis='y', colors='gray')

    # Set grid and spines (borders) color
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#444444')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')

    # Remove the top and right borders for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout to prevent labels from being cut off
    fig.tight_layout()

    # --- Convert plot to Base64 ---
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True, dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig) # Close the figure to free up memory

    return img_base64
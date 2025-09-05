from jinja2 import Environment, FileSystemLoader
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

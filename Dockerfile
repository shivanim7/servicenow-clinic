# Use a lightweight Python image
FROM python:3.11.10-slim

# Set the working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter Notebook and jupyter_server
RUN pip install --upgrade notebook jupyter_server traitlets

# Expose Jupyter Notebook port
EXPOSE 8888

# Command to start Jupyter Notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to prevent .pyc files from being created and to buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create an application directory inside the container
WORKDIR /app

# Copy requirements file to the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the solution files to the container
COPY Solution1.py /app/
COPY Solution2.py /app/
COPY Solution3.py /app/

# Default command to run (can be overridden later)
CMD ["python", "Solution1.py"]

# build.ps1

# Define image name and tar file
$imageName = "flask-docker-app"
$tarFile = "flask-app.tar"

# Step 1: Build the Docker image
Write-Host "🔨 Building Docker image..."
docker build -t $imageName .

# Step 2: Run the Docker container
Write-Host "🚀 Running Docker container..."
docker run -d -p 5000:5000 $imageName

# Step 3: Save the Docker image to a tar file
Write-Host "💾 Saving Docker image to tar file..."
docker save -o $tarFile $imageName

Write-Host "✅ Done! Image saved as $tarFile"

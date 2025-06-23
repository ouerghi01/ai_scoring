# build.ps1

# Define image name and tar file
$imageName = "flask-docker-app"
$tarFile = "flask-app.tar"

# Step 1: Build the Docker image
docker build -t $imageName .

# Step 2: Run the Docker container
docker run -d -p 5000:5000 $imageName

# Step 3: Save the Docker image to a tar file
docker save -o $tarFile $imageName


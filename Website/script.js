document.getElementById("uploadButton").addEventListener("click", async (event) => {
    event.preventDefault(); // Prevent form submission

    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image!");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        // Send the image to the backend
        const response = await fetch("http://127.0.0.1:8000/process-image", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        console.log("RGB Values:", data.rgb_values);

        // Fetch the processed image
        const imageResponse = await fetch("http://127.0.0.1:8000" + data.image_download_url);
        const blob = await imageResponse.blob();
        const imageUrl = URL.createObjectURL(blob);

        // Set the image source to the processed image URL
        const processedImageElement = document.getElementById("processedImage");
        processedImageElement.src = imageUrl;
        processedImageElement.style.display = "block";
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred while processing the image.");
    }
});

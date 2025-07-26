import React, { useState, useRef } from "react";

// Define the types for the component
interface PhotoData {
  id: number;
  name: string;
  imageUrl: string;
  file: File;
}

interface UploadedPhoto {
  file: File;
  preview: string;
  name: string;
}

interface PhotoUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onPhotoUpload: (photo: PhotoData) => void;
}

const PhotoUploadModal: React.FC<PhotoUploadModalProps> = ({
  isOpen,
  onClose,
  onPhotoUpload,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploadedPhoto, setUploadedPhoto] = useState<UploadedPhoto | null>(
    null
  );
  const [photoName, setPhotoName] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle drag events
  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  // Handle dropped files
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  // Handle file selection
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  // Process the selected file
  const handleFile = (file: File) => {
    if (file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        if (e.target && e.target.result) {
          setUploadedPhoto({
            file: file,
            preview: e.target.result as string,
            name: file.name,
          });
          setPhotoName(file.name.split(".")[0]); // Remove extension
        }
      };
      reader.readAsDataURL(file);
    } else {
      alert("Please select an image file.");
    }
  };

  // Handle form submission
  const handleSubmit = async () => {
    if (!uploadedPhoto || !photoName.trim()) return;

    setIsUploading(true);

    // Simulate upload delay
    setTimeout(() => {
      // Call the parent component's upload handler
      onPhotoUpload({
        id: Date.now(), // Generate a simple ID
        name: photoName.trim(),
        imageUrl: uploadedPhoto.preview, // In real app, this would be the uploaded URL
        file: uploadedPhoto.file,
      });

      // Reset form
      setUploadedPhoto(null);
      setPhotoName("");
      setIsUploading(false);
      onClose();
    }, 1500);
  };

  // Reset form when modal closes
  const handleClose = () => {
    setUploadedPhoto(null);
    setPhotoName("");
    setDragActive(false);
    setIsUploading(false);
    onClose();
  };

  // Handle backdrop click
  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      handleClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-opacity-75 flex items-center justify-center z-50 p-4"
      onClick={handleBackdropClick}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="bg-white border border-gray-700 rounded-xl mx-auto w-[30%]"
      >
        {/* Header */}
        <div className="flex justify-between items-center p-6 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-800">Upload Photo</h2>
          <button
            onClick={handleClose}
            className="text-gray-400 hover:text-gray-700 text-2xl leading-none"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Drag and Drop Area */}
          <div
            className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${
              dragActive
                ? "border-blue-400 bg-blue-100 bg-opacity-10"
                : "border-gray-300 hover:border-gray-400"
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            {uploadedPhoto ? (
              <div className="space-y-4">
                <img
                  src={uploadedPhoto.preview}
                  alt="Preview"
                  className="w-32 h-32 object-cover rounded-lg mx-auto border border-gray-300"
                />
                <p className="text-sm text-gray-600">
                  {uploadedPhoto.file.name}
                </p>
                <button
                  type="button"
                  onClick={() => setUploadedPhoto(null)}
                  className="text-sm text-blue-400 hover:text-blue-300"
                >
                  Choose different photo
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="text-4xl">📸</div>
                <div>
                  <p className="text-gray-600 mb-2">
                    Drag and drop your photo here
                  </p>
                  <p className="text-gray-500 text-sm mb-4">or</p>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="bg-white hover:bg-gray-100 text-gray-800 px-4 py-2 rounded-lg border border-gray-300 transition-colors"
                  >
                    Browse Files
                  </button>
                </div>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>

          {/* Photo Name Input */}
          {uploadedPhoto && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Photo Name
              </label>
              <input
                type="text"
                value={photoName}
                onChange={(e) => setPhotoName(e.target.value)}
                placeholder="Enter a name for this photo..."
                className="w-full bg-white border border-gray-300 rounded-lg px-3 py-2 text-gray-800 placeholder-gray-400 focus:border-blue-400 focus:outline-none"
                required
              />
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={handleClose}
              className="flex-1 bg-white hover:bg-gray-100 text-gray-700 py-2 px-4 rounded-lg border border-gray-300 transition-colors"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!uploadedPhoto || !photoName.trim() || isUploading}
              className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:text-gray-500 text-white py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              {isUploading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Uploading...
                </>
              ) : (
                "Upload Photo"
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PhotoUploadModal;

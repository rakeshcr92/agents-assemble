"use client";

import React, { useState, useRef, useEffect } from "react";
import { Swiper, SwiperSlide } from "swiper/react";
import { EffectCoverflow, Pagination, Autoplay } from "swiper/modules";
import type { Swiper as SwiperCore } from "swiper";
import { motion, AnimatePresence } from "framer-motion";
import "swiper/css";
import "swiper/css/effect-coverflow";
import "swiper/css/pagination";

// SHIVANI ADDITION: Importing the photoUpload component
import PhotoUploadModal from "../components/photoUpload";
// Import the voice components and services
import VoiceInterface from "../components/voiceInterface";
import { voiceService, type VoiceQueryResult } from "../services/voiceService";

// Helper component for the word-by-word typing effect
const TypewriterText = ({
  text,
  onComplete,
  className,
}: {
  text: string;
  onComplete?: () => void;
  className?: string;
}) => {
  const [displayedText, setDisplayedText] = useState("");

  useEffect(() => {
    let currentText = "";
    setDisplayedText("");
    const words = text.split(" ");
    let currentWordIndex = 0;
    const intervalId = setInterval(() => {
      if (currentWordIndex < words.length) {
        currentText =
          currentText +
          (currentWordIndex > 0 ? " " : "") +
          words[currentWordIndex];
        setDisplayedText(currentText);
        currentWordIndex++;
      } else {
        clearInterval(intervalId);
        if (onComplete) onComplete();
      }
    }, 120);
    return () => clearInterval(intervalId);
  }, [text, onComplete]);
  return <p className={className}>{displayedText}</p>;
};

export default function Home() {
  const [view, setView] = useState<"home" | "thinking" | "chat">("home");
  const [conversation, setConversation] = useState<
    { sender: "user" | "agent"; query: string; response?: string }[]
  >([]);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isInitiated, setIsInitiated] = useState(false);
  const carouselRef = useRef<SwiperCore | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const chatContainerRef = useRef<HTMLDivElement | null>(null);

  // SHIVANI ADDITION: State to manage the photo upload modal
  const [isPhotoModalOpen, setIsPhotoModalOpen] = useState(false);

  // Voice interface state - UPDATED FOR CONTROLLED MODE
  const [currentUserQuery, setCurrentUserQuery] = useState("");
  const [isVoiceListening, setIsVoiceListening] = useState(false);
  const [isProcessingVoice, setIsProcessingVoice] = useState(false);
  const [voiceTranscript, setVoiceTranscript] = useState("");

  // NEW: Store the actual user queries and control demo vs real mode
  const [actualUserQueries, setActualUserQueries] = useState<string[]>([]);
  const [isUsingRealQueries, setIsUsingRealQueries] = useState(false);

  const memoryPhotos = [
    { id: 1, name: "Jennifer Chen", imageUrl: "/placeholder.png" },
    { id: 2, name: "Jake's Bday", imageUrl: "/placeholder.png" },
    { id: 3, name: "Coffee Meetup", imageUrl: "/placeholder.png" },
    { id: 4, name: "Team Lunch", imageUrl: "/placeholder.png" },
    { id: 5, name: "Project Demo", imageUrl: "/placeholder.png" },
  ];

  useEffect(() => {
    if (chatContainerRef.current) {
      const { scrollHeight, clientHeight } = chatContainerRef.current;
      chatContainerRef.current.scrollTo({
        top: scrollHeight - clientHeight,
        behavior: "smooth",
      });
    }
  }, [conversation]);

  const wait = (ms: number) => new Promise((res) => setTimeout(res, ms));

  // UPDATED: Voice interface handlers with manual control
  const handleVoiceTranscript = async (
    transcript: string,
    isFinal: boolean
  ) => {
    setCurrentUserQuery(transcript);
    setVoiceTranscript(transcript);
  };

  const handleVoiceStart = () => {
    setCurrentUserQuery("");
    setVoiceTranscript("");
  };

  const handleVoiceEnd = () => {
    // Voice ended - handled by controlled state
  };

  const handleVoiceError = (error: string) => {
    console.error("Voice error:", error);
    setIsVoiceListening(false);
    setIsProcessingVoice(false);
  };

  // UPDATED: Toggle voice listening with manual processing
  const toggleVoiceListening = async () => {
    if (isProcessingVoice) return;

    if (isVoiceListening) {
      // User is stopping the recording
      setIsVoiceListening(false);

      // Process the accumulated transcript if we have any
      if (voiceTranscript.trim()) {
        setIsProcessingVoice(true);

        try {
          // Process the voice query using the voice service
          const result: VoiceQueryResult = await voiceService.processVoiceQuery(
            voiceTranscript
          );

          if (result.success) {
            await handleVoiceQuery(voiceTranscript, result.response);
          } else {
            setCurrentUserQuery(
              "I'm sorry, I couldn't process your request. Please try again."
            );
          }
        } catch (error) {
          console.error("Error processing voice query:", error);
          setCurrentUserQuery(
            "I'm experiencing some technical difficulties. Please try again."
          );
        } finally {
          setIsProcessingVoice(false);
        }
      }
    } else {
      // User is starting to record
      setIsVoiceListening(true);
    }
  };

  // Handle voice query processing
  const handleVoiceQuery = async (query: string, response: string) => {
    if (view === "home") {
      // If on home page, start the conversation with the voice query
      setIsInitiated(true);
      setIsUsingRealQueries(true);
      setActualUserQueries([query]);
      if (carouselRef.current) carouselRef.current.autoplay.stop();

      // Simulate thinking phase
      setView("thinking");
      await wait(2000);

      // Move to chat with the voice query
      setConversation([{ sender: "user", query: query }]);
      setView("chat");

      // Wait for user query typewriter to complete before showing response
      // Calculate approximate time for user query to finish typing (120ms per word + buffer)
      const words = query.split(" ");
      const typewriterDuration = words.length * 120 + 500; // 500ms buffer
      await wait(typewriterDuration);

      // Add response after user query is fully displayed
      setConversation((prev) => [
        ...prev,
        { sender: "agent", query: "", response: response },
      ]);
      setCurrentStepIndex(1);
    } else if (view === "chat") {
      // If already in chat, add the new query to conversation
      const newQueryIndex = actualUserQueries.length;
      setActualUserQueries((prev) => [...prev, query]);
      setConversation((prev) => [...prev, { sender: "user", query: query }]);

      // Wait for user query typewriter to complete before showing response
      const words = query.split(" ");
      const typewriterDuration = words.length * 120 + 500; // 500ms buffer
      await wait(typewriterDuration);

      // Add response after user query is fully displayed
      setConversation((prev) => [
        ...prev,
        { sender: "agent", query: "", response: response },
      ]);
      setCurrentStepIndex((prev) => prev + 1);
    }
  };

  const switchToThinking = async () => {
    await wait(1000);
    setView("thinking");
    await wait(2000);

    // Use the first user query
    const firstQuery = actualUserQueries[0];
    setConversation([{ sender: "user", query: firstQuery }]);
    setView("chat");

    // Wait for user query typewriter to complete before showing response
    const words = firstQuery.split(" ");
    const typewriterDuration = words.length * 120 + 500; // 500ms buffer
    await wait(typewriterDuration);

    // Add a placeholder response (you can modify this based on your needs)
    const firstResponse = "I'm processing your memory request...";
    setConversation((prev) => [
      ...prev,
      { sender: "agent", query: "", response: firstResponse },
    ]);
  };

  const handleGoHome = () => {
    setView("home");
    setConversation([]);
    setCurrentStepIndex(0);
    setIsInitiated(false);
    setIsUsingRealQueries(false);
    setActualUserQueries([]);
    setCurrentUserQuery("");
    setVoiceTranscript("");
    setIsVoiceListening(false);
    setIsProcessingVoice(false);
    voiceService.resetConversation();
    if (carouselRef.current) {
      carouselRef.current.autoplay?.start();
    }
  };

  return (
    <div className="bg-gray-50 text-gray-800 min-h-screen font-sans overflow-hidden relative">
      {/* This div blurs the background when the modal is open */}
      <div
        className={`transition-all duration-300 ${
          isPhotoModalOpen ? "blur-sm" : ""
        }`}
      >
        <AnimatePresence>
          {view === "home" && (
            <motion.div
              initial={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
              className="absolute inset-0"
            >
              <div className="bg-black">
                <header className="flex justify-between items-center p-5 max-w-5xl mx-auto">
                  <div className="text-2xl font-bold text-white">
                    Life Witness
                  </div>
                  <nav className="flex gap-4 sm:gap-6 text-white">
                    <a href="/our-project" className="hover:text-gray-300">
                      Our Project
                    </a>
                    <a href="/meet-your-agents" className="hover:text-gray-300">
                      Meet Your Agents
                    </a>
                    <a href="/settings" className="hover:text-gray-300">
                      Settings
                    </a>
                  </nav>
                </header>
              </div>
              <div className="p-5 sm:p-8 flex flex-col h-full max-w-5xl mx-auto">
                <main className="flex-grow flex flex-col items-center justify-center space-y-10">
                  <div className="text-center">
                    <h1 className="text-5xl font-bold text-gray-800">
                      <span className="bg-gradient-to-r from-amber-500 to-orange-500 bg-clip-text text-transparent">
                        Hello,
                      </span>{" "}
                      what memory
                    </h1>
                    <h1 className="text-5xl font-bold text-gray-800 mt-2">
                      would you like to relive today?
                    </h1>
                  </div>
                  <Swiper
                    onSwiper={(swiper) => {
                      carouselRef.current = swiper;
                    }}
                    onSlideChange={(swiper) => setActiveIndex(swiper.realIndex)}
                    effect={"coverflow"}
                    grabCursor={true}
                    centeredSlides={true}
                    slidesPerView={3}
                    loop={true}
                    autoplay={{ delay: 3000, disableOnInteraction: false }}
                    modules={[EffectCoverflow, Pagination, Autoplay]}
                    className="w-full"
                    coverflowEffect={{
                      rotate: 0,
                      stretch: 80,
                      depth: 150,
                      modifier: 1,
                      slideShadows: false,
                    }}
                  >
                    {memoryPhotos.map((photo, index) => (
                      <SwiperSlide key={photo.id}>
                        <div className="flex flex-col items-center justify-center h-full pt-10">
                          <img
                            src={photo.imageUrl}
                            alt={photo.name || ""}
                            className={`w-32 h-32 sm:w-40 sm:h-40 object-cover rounded-2xl shadow-lg transition-all duration-300 ${
                              activeIndex === index
                                ? "scale-110 shadow-2xl"
                                : "scale-90 opacity-60"
                            }`}
                          />
                          <span
                            className={`block mt-4 font-medium text-sm transition-opacity duration-300 ${
                              activeIndex === index
                                ? "opacity-100"
                                : "opacity-60"
                            }`}
                          >
                            {photo.name}
                          </span>
                        </div>
                      </SwiperSlide>
                    ))}
                  </Swiper>
                  <div className="w-full max-w-3xl bg-white p-2 rounded-full border border-gray-200 shadow-sm flex items-center h-20">
                    <div className="flex-grow px-4">
                      {isInitiated ? (
                        <TypewriterText
                          text={actualUserQueries[0] || ""}
                          onComplete={switchToThinking}
                          className="text-gray-800 text-base"
                        />
                      ) : (
                        <p className="text-gray-500 text-base">
                          {isProcessingVoice
                            ? "Processing your request..."
                            : isVoiceListening && voiceTranscript
                            ? voiceTranscript
                            : currentUserQuery || "Ask about a memory..."}
                        </p>
                      )}
                    </div>
                    {/* UPDATED: Use controlled VoiceInterface */}
                    <VoiceInterface
                      onTranscript={handleVoiceTranscript}
                      onVoiceStart={handleVoiceStart}
                      onVoiceEnd={handleVoiceEnd}
                      onError={handleVoiceError}
                      disabled={isProcessingVoice}
                      isListening={isVoiceListening}
                      onToggleListening={toggleVoiceListening}
                    />
                    {/* --- PARTNER'S ADDITION: onClick handler for the upload button --- */}
                    <button
                      onClick={() => setIsPhotoModalOpen(true)}
                      className="bg-blue-500 hover:bg-blue-600 text-white rounded-full py-3 px-6 font-semibold text-sm"
                    >
                      Upload Memory
                    </button>
                  </div>
                </main>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {view === "thinking" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 flex flex-col items-center justify-center bg-white"
            >
              <p className="text-lg mb-4">"{actualUserQueries[0] || ""}"</p>
              <div className="flex items-center justify-center gap-2">
                <motion.span
                  animate={{ y: [0, -10, 0] }}
                  transition={{
                    duration: 1.2,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }}
                  className="w-3 h-3 bg-blue-500 rounded-full"
                />
                <motion.span
                  animate={{ y: [0, -10, 0] }}
                  transition={{
                    duration: 1.2,
                    repeat: Infinity,
                    ease: "easeInOut",
                    delay: 0.2,
                  }}
                  className="w-3 h-3 bg-blue-500 rounded-full"
                />
                <motion.span
                  animate={{ y: [0, -10, 0] }}
                  transition={{
                    duration: 1.2,
                    repeat: Infinity,
                    ease: "easeInOut",
                    delay: 0.4,
                  }}
                  className="w-3 h-3 bg-blue-500 rounded-full"
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {view === "chat" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
              className="absolute inset-0"
            >
              <div className="flex h-screen">
                <div className="w-1/3 bg-gray-50 border-r border-gray-200 p-8 flex flex-col justify-center items-center relative">
                  <button
                    onClick={handleGoHome}
                    className="absolute top-6 left-6 text-2xl hover:text-gray-500"
                  >
                    &times;
                  </button>
                  <div className="text-center">
                    <img
                      src="/placeholder.png"
                      alt="Jennifer Chen"
                      className="w-52 h-52 object-cover rounded-xl shadow-xl"
                    />
                    <p className="mt-4 text-lg font-semibold">Jennifer Chen</p>
                  </div>
                </div>
                <div className="w-2/3 flex flex-col p-8 bg-white">
                  <div
                    ref={chatContainerRef}
                    className="flex-grow overflow-y-auto pr-4 space-y-6"
                  >
                    {conversation.map((chat, index) => (
                      <div key={index}>
                        {chat.sender === "user" ? (
                          <div className="flex items-start gap-3 justify-end">
                            <div className="bg-blue-500 text-white p-3 rounded-lg max-w-xl">
                              <TypewriterText text={chat.query} />
                            </div>
                          </div>
                        ) : (
                          <div className="flex items-start gap-3">
                            <div className="bg-gray-100 p-3 rounded-lg max-w-xl">
                              <TypewriterText text={chat.response || ""} />
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>

                  {/* Chat input area */}
                  <div className="flex-shrink-0 pt-6">
                    {/* Voice interface */}
                    <div className="flex justify-center">
                      <VoiceInterface
                        onTranscript={handleVoiceTranscript}
                        onVoiceStart={handleVoiceStart}
                        onVoiceEnd={handleVoiceEnd}
                        onError={handleVoiceError}
                        disabled={isProcessingVoice}
                        isListening={isVoiceListening}
                        onToggleListening={toggleVoiceListening}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* SHIVANI ADDITION: The modal component itself */}
      <PhotoUploadModal
        isOpen={isPhotoModalOpen}
        onClose={() => setIsPhotoModalOpen(false)}
        onPhotoUpload={(photo) => {
          // This is where you'd handle the uploaded photo data
          console.log("New photo uploaded:", photo);
          setIsPhotoModalOpen(false); // Close modal after upload
        }}
      />
    </div>
  );
}

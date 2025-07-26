"use client";

import React, { useState, useRef, useEffect } from "react";

interface VoiceInterfaceProps {
  onTranscript: (transcript: string, isFinal: boolean) => void;
  onVoiceStart: () => void;
  onVoiceEnd: () => void;
  onError: (error: string) => void;
  disabled?: boolean;
  // New props for external control
  isListening?: boolean;
  onToggleListening?: () => void;
}

const VoiceInterface: React.FC<VoiceInterfaceProps> = ({
  onTranscript,
  onVoiceStart,
  onVoiceEnd,
  onError,
  disabled = false,
  isListening: externalIsListening,
  onToggleListening,
}) => {
  const [internalIsListening, setInternalIsListening] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  // Add accumulated transcript to handle pauses
  const accumulatedTranscriptRef = useRef<string>("");
  const lastFinalTranscriptRef = useRef<string>("");

  // Use external control if provided, otherwise fall back to internal state
  const isListening =
    externalIsListening !== undefined
      ? externalIsListening
      : internalIsListening;
  const controlled = externalIsListening !== undefined;

  useEffect(() => {
    // Check if speech recognition is supported
    if (typeof window !== "undefined") {
      const SpeechRecognitionClass =
        window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognitionClass) {
        setIsSupported(true);

        const recognition = new SpeechRecognitionClass();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "en-US";

        recognition.onstart = () => {
          console.log("Speech recognition started");
          // Reset accumulated transcript when starting
          accumulatedTranscriptRef.current = "";
          lastFinalTranscriptRef.current = "";

          if (!controlled) {
            setInternalIsListening(true);
          }
          onVoiceStart();
        };

        recognition.onresult = (event: SpeechRecognitionEvent) => {
          let interimTranscript = "";
          let finalTranscript = "";

          // Process all results from the last processed index
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              finalTranscript += transcript;
            } else {
              interimTranscript += transcript;
            }
          }

          // Handle final results - accumulate them
          if (finalTranscript) {
            // Only add new final transcript (avoid duplicates)
            if (finalTranscript !== lastFinalTranscriptRef.current) {
              accumulatedTranscriptRef.current +=
                (accumulatedTranscriptRef.current ? " " : "") +
                finalTranscript.trim();
              lastFinalTranscriptRef.current = finalTranscript;
            }
          }

          // Combine accumulated final transcript with current interim
          const fullTranscript =
            accumulatedTranscriptRef.current +
            (accumulatedTranscriptRef.current && interimTranscript ? " " : "") +
            interimTranscript;

          // Send the complete transcript (accumulated + interim)
          if (fullTranscript.trim()) {
            onTranscript(fullTranscript.trim(), !interimTranscript);
          }
        };

        recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
          console.error("Speech recognition error:", event.error);

          // Handle "aborted" specifically - don't show as error to user
          if (event.error === "aborted") {
            console.log(
              "Speech recognition was aborted - this is normal behavior"
            );
            if (!controlled) {
              setInternalIsListening(false);
            }
            onVoiceEnd();
            return;
          }

          // Handle "no-speech" - also normal, but restart recognition if still listening
          if (event.error === "no-speech") {
            console.log("No speech detected");
            // If we're still supposed to be listening, restart recognition
            if (isListening && !disabled) {
              console.log("Restarting recognition due to no-speech...");
              setTimeout(() => {
                if (
                  recognitionRef.current &&
                  (controlled ? externalIsListening : internalIsListening)
                ) {
                  try {
                    recognitionRef.current.start();
                  } catch (restartError) {
                    console.error(
                      "Failed to restart after no-speech:",
                      restartError
                    );
                  }
                }
              }, 100);
            } else {
              if (!controlled) {
                setInternalIsListening(false);
              }
              onVoiceEnd();
            }
            return;
          }

          // Only show actual errors to user
          if (!["aborted", "no-speech"].includes(event.error)) {
            if (!controlled) {
              setInternalIsListening(false);
            }
            onError(`Speech recognition error: ${event.error}`);
            onVoiceEnd();
          }
        };

        recognition.onend = () => {
          console.log("Speech recognition ended");

          // If we're still supposed to be listening, restart recognition
          const shouldStillListen = controlled
            ? externalIsListening
            : internalIsListening;

          if (shouldStillListen && !disabled) {
            console.log(
              "Recognition ended but should still be listening, restarting..."
            );
            setTimeout(() => {
              if (recognitionRef.current && shouldStillListen) {
                try {
                  recognitionRef.current.start();
                } catch (restartError) {
                  console.error("Failed to restart recognition:", restartError);
                  if (!controlled) {
                    setInternalIsListening(false);
                  }
                  onVoiceEnd();
                }
              }
            }, 100);
          } else {
            if (!controlled) {
              setInternalIsListening(false);
            }
            onVoiceEnd();
          }
        };

        recognitionRef.current = recognition;
      } else {
        setIsSupported(false);
      }
    }
  }, [
    onTranscript,
    onVoiceStart,
    onVoiceEnd,
    onError,
    controlled,
    internalIsListening,
    externalIsListening,
    disabled,
  ]);

  // Effect to handle external listening state changes
  useEffect(() => {
    if (controlled && recognitionRef.current) {
      if (externalIsListening && !internalIsListening) {
        startListening();
      } else if (!externalIsListening && internalIsListening) {
        stopListening();
      }
    }
  }, [externalIsListening, controlled, internalIsListening]);

  const startListening = () => {
    if (recognitionRef.current && !disabled) {
      // Reset transcript accumulation
      accumulatedTranscriptRef.current = "";
      lastFinalTranscriptRef.current = "";

      try {
        setInternalIsListening(true);
        recognitionRef.current.start();
      } catch (error: any) {
        console.error("Error starting speech recognition:", error);

        // If recognition is already started, try to stop and restart
        if (error.name === "InvalidStateError") {
          console.log("Recognition already started, attempting restart...");
          try {
            recognitionRef.current.stop();
            setTimeout(() => {
              if (recognitionRef.current) {
                recognitionRef.current.start();
              }
            }, 200);
          } catch (restartError) {
            console.error("Failed to restart recognition:", restartError);
            setInternalIsListening(false);
            onError("Failed to start speech recognition");
          }
        } else {
          setInternalIsListening(false);
          onError("Failed to start speech recognition");
        }
      }
    }
  };

  const stopListening = () => {
    if (recognitionRef.current) {
      setInternalIsListening(false);
      recognitionRef.current.stop();
    }
  };

  const handleToggle = () => {
    if (controlled && onToggleListening) {
      onToggleListening();
    } else {
      // Uncontrolled mode
      if (isListening) {
        stopListening();
      } else {
        startListening();
      }
    }
  };

  if (!isSupported) {
    return (
      <div className="flex items-center justify-center">
        <div className="bg-red-100 text-red-700 px-4 py-2 rounded-lg text-sm">
          Speech recognition not supported in this browser
        </div>
      </div>
    );
  }

  return (
    <button
      onClick={handleToggle}
      disabled={disabled}
      className={`
        rounded-full p-3 text-2xl flex items-center justify-center disabled:opacity-50 mx-2
        ${
          isListening
            ? "bg-red-100 hover:bg-red-200 animate-pulse"
            : "bg-gray-100 hover:bg-gray-200"
        }
      `}
      title={
        isListening ? "Click to stop recording" : "Click to start recording"
      }
    >
      {isListening ? (
        <div className="flex items-center justify-center">
          <div className="w-3 h-3 bg-red-500 rounded-full animate-ping"></div>
        </div>
      ) : (
        "🎤"
      )}
    </button>
  );
};

export default VoiceInterface;

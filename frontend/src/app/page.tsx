'use client'; 

import React, { useState, useRef, useEffect } from 'react';
import { Swiper, SwiperSlide } from 'swiper/react';
import { EffectCoverflow, Pagination, Autoplay } from 'swiper/modules';
import type { Swiper as SwiperCore } from 'swiper';
import { motion, AnimatePresence } from 'framer-motion';
import 'swiper/css';
import 'swiper/css/effect-coverflow';
import 'swiper/css/pagination';

export default function Home() {
  // ----------------------------------------------------------------
  // --- 1. STATE MANAGEMENT ---
  // All the variables that store information for the page.
  // ----------------------------------------------------------------
  const [view, setView] = useState<'home' | 'thinking' | 'chat'>('home'); 
  const [conversation, setConversation] = useState<{ sender: 'user' | 'agent'; query: string; response?: string }[]>([]);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isInitiated, setIsInitiated] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const carouselRef = useRef<SwiperCore | null>(null);
  const chatContainerRef = useRef<HTMLDivElement | null>(null);

  // 2. DEMO DATA
  // All the hard-coded content for the prototype.
  const demoConversation = [ { query: "Who was that person I met at TechCrunch who worked on crypto payments?", response: "That would be Jennifer Chen, VP of Engineering at Stripe! You met her at their booth on day 2 of TechCrunch conference. She mentioned they were hiring senior engineers for their crypto payments team." }, { query: "Right! What else did we talk about?", response: "You both bonded over being Stanford alumni, and she specifically mentioned their new payment APIs. You took a photo together at their booth, and she gave you her business card with instructions to reach out about the senior engineering role." }, { query: "Did I ever follow up with her?", response: "Yes, I reminded you on Tuesday morning as requested, and you sent her a LinkedIn message that same day. She responded within 2 hours suggesting a coffee meeting, which you scheduled for the next Friday. You met at Blue Bottle Coffee downtown and had what you described as a 'great conversation about team culture and technical challenges.'" }, { query: "Perfect. I'm interviewing at Coinbase tomorrow and want to mention my connection to Stripe's crypto work. Can you give me some talking points?", response: "Based on your conversation with Jennifer, here are some relevant points: • Stripe is actively investing in crypto infrastructure with dedicated teams • They're focusing on enterprise-grade payment solutions (Jennifer's emphasis) • Their approach prioritizes regulatory compliance and security • Jennifer mentioned they're seeing huge demand from fintech companies. This shows you're plugged into industry trends and have insights from a senior leader in the space. Would you like me to find other crypto-related conversations from your memory?" } ];
  const memoryPhotos = [ { id: 1, name: 'Jennifer Chen', imageUrl: '/placeholder.png' }, { id: 2, name: 'Jake\'s Bday', imageUrl: '/placeholder.png' }, { id: 3, name: 'Coffee Meetup', imageUrl: '/placeholder.png' }, { id: 4, name: 'Team Lunch', imageUrl: '/placeholder.png' }, { id: 5, name: 'Project Demo', imageUrl: '/placeholder.png' } ];

  // 3. HELPER FUNCTIONS
  // Reusable logic, like creating a delay.
  const wait = (ms: number) => new Promise(res => setTimeout(res, ms));

  useEffect(() => {
    if (chatContainerRef.current) {
      const { scrollHeight, clientHeight } = chatContainerRef.current;
      chatContainerRef.current.scrollTo({ top: scrollHeight - clientHeight, behavior: 'smooth' });
    }
  }, [conversation]);

  // 4. EVENT HANDLERS
  // All the functions that run when the user does something.
  const handleInitiateConversation = async () => {
    if (view !== 'home' || currentStepIndex !== 0) return;
    setIsInitiated(true);
    if (carouselRef.current) carouselRef.current.autoplay.stop();
  };

  const switchToThinking = async () => {
    await wait(1000);
    setView('thinking');
    await wait(2000); 
    const firstStep = demoConversation[0];
    setConversation([{ sender: 'user', query: firstStep.query }]);
    setCurrentStepIndex(1);
    setView('chat');
    await wait(2500); 
    setConversation(prev => [...prev, { sender: 'agent', query: '', response: firstStep.response }]);
  };

  const runNextStepInDemo = async () => {
    const currentStep = demoConversation[currentStepIndex];
    if (!currentStep) return;
    setConversation(prev => [ ...prev, { sender: 'user', query: currentStep.query }]);
    await wait(1500);
    setConversation(prev => [ ...prev, { sender: 'agent', query: '', response: currentStep.response } ]);
    setCurrentStepIndex(prev => (prev + 1) % demoConversation.length);
  };
  
  const handleGoHome = () => {
      setView('home');
      setConversation([]);
      setCurrentStepIndex(0);
      setIsInitiated(false);
      if (carouselRef.current) {
        carouselRef.current.autoplay?.start();
      }
  };
  
  // 5. HELPER COMPONENTS ---
  // Small, reusable components used only on this page.
  const TypewriterText = ({ text, onComplete, className }: { text: string, onComplete?: () => void, className?: string }) => {
    const [displayedText, setDisplayedText] = useState('');
    useEffect(() => {
      let currentText = '';
      setDisplayedText('');
      const words = text.split(' ');
      let currentWordIndex = 0;
      const intervalId = setInterval(() => {
        if (currentWordIndex < words.length) {
          currentText += (currentWordIndex > 0 ? ' ' : '') + words[currentWordIndex];
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

  // --- 6. RENDER ---
  // The actual JSX that gets rendered to the screen.
  return (
    <div className="bg-gray-50 text-gray-800 min-h-screen font-sans overflow-hidden relative">
      <AnimatePresence>
        {view === 'home' && (
          <motion.div initial={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.5 }} className="absolute inset-0">
            <div className="bg-black">
              <header className="flex justify-between items-center p-5 max-w-5xl mx-auto">
                  <div className="text-2xl font-bold text-white">Life Witness</div>
                  <nav className="flex gap-4 sm:gap-6 text-gray-400">
                      <a href="/our-project" className="hover:text-white">Our Project</a>
                      <a href="/meet-your-agents" className="hover:text-white">Meet Your Agents</a>
                      <a href="/settings" className="hover:text-white">Settings</a>
                  </nav>
              </header>
            </div>
            <div className="p-5 sm:p-8 flex flex-col h-full max-w-5xl mx-auto">
                <main className="flex-grow flex flex-col items-center justify-center space-y-10">
                    <div className="text-center">
                        <h1 className="text-5xl font-bold text-gray-800">
                            <span className="bg-gradient-to-r from-amber-500 to-orange-500 bg-clip-text text-transparent">Hello,</span> what memory
                        </h1>
                        <h1 className="text-5xl font-bold text-gray-800 mt-2">
                            would you like to relive today?
                        </h1>
                    </div>
                    <Swiper 
                      onSwiper={(swiper) => { carouselRef.current = swiper; }} 
                      onSlideChange={(swiper) => setActiveIndex(swiper.realIndex)}
                      effect={'coverflow'} grabCursor={true} centeredSlides={true} 
                      slidesPerView={3} loop={true} autoplay={{ delay: 3000, disableOnInteraction: false }} 
                      modules={[EffectCoverflow, Pagination, Autoplay]} className="w-full"
                      coverflowEffect={{ rotate: 0, stretch: 80, depth: 150, modifier: 1, slideShadows: false }}
                    >
                        {memoryPhotos.map((photo, index) => (
                            <SwiperSlide key={photo.id}>
                              <div className="flex flex-col items-center justify-center h-full pt-10">
                                <img src={photo.imageUrl} alt={photo.name || ''} className={`w-32 h-32 sm:w-40 sm:h-40 object-cover rounded-2xl shadow-lg transition-all duration-300 ${activeIndex === index ? 'scale-110 shadow-2xl' : 'scale-90 opacity-60'}`}/>
                                <span className={`block mt-4 font-medium text-sm transition-opacity duration-300 ${activeIndex === index ? 'opacity-100' : 'opacity-60'}`}>{photo.name}</span>
                              </div>
                            </SwiperSlide>
                        ))}
                    </Swiper>
                    <div className="w-full max-w-3xl bg-white p-2 rounded-full border border-gray-200 shadow-sm flex items-center h-20">
                        <div className="flex-grow px-4">
                          {isInitiated ? (
                              <TypewriterText text={`${demoConversation[0].query}`} onComplete={switchToThinking} className="text-gray-800 text-base" />
                          ) : (
                              <p className="text-gray-500 text-base">Ask about a memory...</p>
                          )}
                        </div>
                        <button onClick={handleInitiateConversation} disabled={isInitiated} className="bg-gray-100 hover:bg-gray-200 rounded-full p-3 text-2xl flex items-center justify-center disabled:opacity-50 mx-2">🎤</button>
                        <button className="bg-blue-500 hover:bg-blue-600 text-white rounded-full py-3 px-6 font-semibold text-sm">Upload Memory</button>
                    </div>
                </main>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {view === 'thinking' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 flex flex-col items-center justify-center bg-white">
            <p className="text-lg mb-4">"{demoConversation[0].query}"</p>
            <div className="flex items-center justify-center gap-2">
              <motion.span animate={{ y: [0, -10, 0] }} transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }} className="w-3 h-3 bg-blue-500 rounded-full"/>
              <motion.span animate={{ y: [0, -10, 0] }} transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut", delay: 0.2 }} className="w-3 h-3 bg-blue-500 rounded-full"/>
              <motion.span animate={{ y: [0, -10, 0] }} transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut", delay: 0.4 }} className="w-3 h-3 bg-blue-500 rounded-full"/>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {view === 'chat' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }} className="absolute inset-0">
            <div className="flex h-screen">
              <div className="w-1/3 bg-gray-50 border-r border-gray-200 p-8 flex flex-col justify-center items-center relative">
                <button onClick={handleGoHome} className="absolute top-6 left-6 text-2xl hover:text-gray-500">&times;</button>
                <div className="text-center"><img src="/placeholder.png" alt="Jennifer Chen" className="w-52 h-52 object-cover rounded-xl shadow-xl"/><p className="mt-4 text-lg font-semibold">Jennifer Chen</p></div>
              </div>
              <div className="w-2/3 flex flex-col p-8 bg-white">
                <div ref={chatContainerRef} className="flex-grow overflow-y-auto pr-4 space-y-6">
                  {conversation.map((chat, index) => (<div key={index}>{chat.sender === 'user' ? (<div className="flex items-start gap-3 justify-end"><div className="bg-blue-500 text-white p-3 rounded-lg max-w-xl"><TypewriterText text={chat.query} /></div></div>) : (<div className="flex items-start gap-3"><div className="bg-gray-100 p-3 rounded-lg max-w-xl"><TypewriterText text={chat.response || ''} /></div></div>)}</div>))}
                </div>
                <div className="flex-shrink-0 pt-6 flex justify-center">
                    <button onClick={runNextStepInDemo} className="bg-gray-200 hover:bg-gray-300 rounded-full w-16 h-16 text-4xl flex items-center justify-center">🎤</button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
// VoiceService.ts - Mock service for handling voice queries

export interface VoiceQueryResult {
  success: boolean;
  response: string;
  matchedMemoryId?: number;
  error?: string;
}

class VoiceService {
  private demoConversation = [
    {
      query: "Who was that person I met at TechCrunch who worked on crypto payments?",
      response: "That would be Jennifer Chen, VP of Engineering at Stripe! You met her at their booth on day 2 of TechCrunch conference. She mentioned they were hiring senior engineers for their crypto payments team.",
      memoryId: 1
    },
    {
      query: "Right! What else did we talk about?",
      response: "You both bonded over being Stanford alumni, and she specifically mentioned their new payment APIs. You took a photo together at their booth, and she gave you her business card with instructions to reach out about the senior engineering role.",
      memoryId: 1
    },
    {
      query: "Did I ever follow up with her?",
      response: "Yes, I reminded you on Tuesday morning as requested, and you sent her a LinkedIn message that same day. She responded within 2 hours suggesting a coffee meeting, which you scheduled for the next Friday. You met at Blue Bottle Coffee downtown and had what you described as a 'great conversation about team culture and technical challenges.'",
      memoryId: 1
    },
    {
      query: "Perfect. I'm interviewing at Coinbase tomorrow and want to mention my connection to Stripe's crypto work. Can you give me some talking points?",
      response: "Based on your conversation with Jennifer, here are some relevant points: • Stripe is actively investing in crypto infrastructure with dedicated teams • They're focusing on enterprise-grade payment solutions (Jennifer's emphasis) • Their approach prioritizes regulatory compliance and security • Jennifer mentioned they're seeing huge demand from fintech companies. This shows you're plugged into industry trends and have insights from a senior leader in the space. Would you like me to find other crypto-related conversations from your memory?",
      memoryId: 1
    },
    {
      query: "Tell me about Jake's birthday party",
      response: "Jake's birthday party was last month at his apartment downtown. You brought homemade cupcakes and played board games until midnight. Sarah was there too, and you all had a great time catching up. Jake mentioned he's planning to move to Seattle for a new job opportunity.",
      memoryId: 2
    },
    {
      query: "What happened at the coffee meetup?",
      response: "At the coffee meetup last week, you discussed the new startup ideas with Mike and Lisa. You sketched out some wireframes on napkins and talked about potential funding sources. Lisa offered to introduce you to her contact at Y Combinator.",
      memoryId: 3
    },
    {
      query: "Remind me about the team lunch",
      response: "The team lunch was at that Italian restaurant near the office. Everyone celebrated Tom's promotion, and you discussed the Q4 roadmap. The pasta was excellent, and you all decided to make it a monthly tradition.",
      memoryId: 4
    },
    {
      query: "What was the project demo about?",
      response: "The project demo showcased your new AI-powered analytics dashboard. The stakeholders were impressed with the real-time data visualization features. You received approval to move forward with the full implementation, and the VP mentioned it could be a game-changer for the company.",
      memoryId: 5
    }
  ];

  private currentConversationIndex = 0;
  private usedQueries = new Set<string>();

  // Simulate processing delay like a real API
  private async simulateDelay(min: number = 800, max: number = 2000): Promise<void> {
    const delay = Math.random() * (max - min) + min;
    return new Promise(resolve => setTimeout(resolve, delay));
  }

  // Find the best matching response for the user's query
  private findBestMatch(userQuery: string): { response: string; memoryId?: number } {
    const queryLower = userQuery.toLowerCase();
    
    // Keywords mapping for better matching
    const keywordMatches = {
      'jennifer': [0, 1, 2, 3],
      'chen': [0, 1, 2, 3],
      'stripe': [0, 1, 2, 3],
      'techcrunch': [0],
      'crypto': [0, 3],
      'jake': [4],
      'birthday': [4],
      'coffee': [5],
      'meetup': [5],
      'lunch': [6],
      'team': [6],
      'demo': [7],
      'project': [7],
      'coinbase': [3],
      'interview': [3]
    };

    // Find matches based on keywords
    let bestMatches: number[] = [];
    for (const [keyword, indices] of Object.entries(keywordMatches)) {
      if (queryLower.includes(keyword)) {
        bestMatches = [...bestMatches, ...indices];
      }
    }

    // Remove duplicates and sort
    bestMatches = [...new Set(bestMatches)];

    // If we have keyword matches, use the first unused one
    for (const index of bestMatches) {
      const item = this.demoConversation[index];
      if (!this.usedQueries.has(item.query)) {
        this.usedQueries.add(item.query);
        return { response: item.response, memoryId: item.memoryId };
      }
    }

    // If no keyword matches or all used, use the next in sequence
    if (this.currentConversationIndex < this.demoConversation.length) {
      const item = this.demoConversation[this.currentConversationIndex];
      this.currentConversationIndex++;
      return { response: item.response, memoryId: item.memoryId };
    }

    // Fallback response
    return {
      response: "I'm searching through your memories for that information. Could you provide more context or try asking about a specific person, event, or topic?",
      memoryId: undefined
    };
  }

  // Main method to process voice queries
  async processVoiceQuery(userQuery: string): Promise<VoiceQueryResult> {
    try {
      // Simulate processing time
      await this.simulateDelay();

      // Clean up the query
      const cleanQuery = userQuery.trim();
      
      if (!cleanQuery) {
        return {
          success: false,
          response: "I didn't catch that. Could you please repeat your question?",
          error: "Empty query"
        };
      }

      // Find the best matching response
      const match = this.findBestMatch(cleanQuery);

      return {
        success: true,
        response: match.response,
        matchedMemoryId: match.memoryId
      };

    } catch (error) {
      console.error('Error processing voice query:', error);
      return {
        success: false,
        response: "I'm sorry, I encountered an error processing your request. Please try again.",
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  // Reset the conversation state
  resetConversation(): void {
    this.currentConversationIndex = 0;
    this.usedQueries.clear();
  }

  // Get all available demo queries (useful for testing)
  getDemoQueries(): string[] {
    return this.demoConversation.map(item => item.query);
  }
}

// Export a singleton instance
export const voiceService = new VoiceService();
export default VoiceService;
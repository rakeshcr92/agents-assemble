export interface VoiceQuery {
  query: string;
  response: string;
  memoryId: number;
}

export interface VoiceQueryResult {
  success: boolean;
  response?: string;
  memoryId?: number;
  error?: string;
}

export interface ConversationContext {
  currentMemoryId?: number;
  conversationHistory: VoiceQuery[];
  memorySequenceIndex: { [memoryId: number]: number };
}

class VoiceService {
  private demoConversation: VoiceQuery[] = [
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
      response: "Yes, I reminded you on Tuesday morning as requested, and you sent her a LinkedIn message that same day. She responded within 2 hours suggesting a coffee meeting, which you scheduled for the next Friday. You met at Blue Bottle Coffee downtown and had what you described as a 'great conversation about team culture and technical challenges.",
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

  private conversationContext: ConversationContext = {
    conversationHistory: [],
    memorySequenceIndex: {}
  };

  /**
   * Simulates processing a voice query and returns a response
   * @param query - The transcribed text from speech
   * @returns Promise with the response data
   */
  async processVoiceQuery(query: string): Promise<VoiceQueryResult> {
    try {
      // Simulate network delay
      await this.delay(500 + Math.random() * 1000);

      // Check for topic triggers and get sequential response
      const response = this.getSequentialResponse(query);
      
      if (response) {
        // Add to conversation history
        this.conversationContext.conversationHistory.push(response);
        this.conversationContext.currentMemoryId = response.memoryId;

        return {
          success: true,
          response: response.response,
          memoryId: response.memoryId
        };
      } else {
        // Fallback response for unmatched queries
        const fallbackResponse = this.generateFallbackResponse(query);
        const fallbackQuery: VoiceQuery = {
          query,
          response: fallbackResponse,
          memoryId: 0
        };
        
        this.conversationContext.conversationHistory.push(fallbackQuery);

        return {
          success: true,
          response: fallbackResponse,
          memoryId: 0
        };
      }
    } catch (error) {
      return {
        success: false,
        error: 'Failed to process voice query'
      };
    }
  }

  /**
   * Get conversation history
   * @returns Array of previous queries and responses
   */
  getConversationHistory(): VoiceQuery[] {
    return this.conversationContext.conversationHistory;
  }

  /**
   * Clear conversation history and reset sequence indices
   */
  clearConversationHistory(): void {
    this.conversationContext = {
      conversationHistory: [],
      memorySequenceIndex: {}
    };
  }

  /**
   * Reset conversation (alias for clearConversationHistory for backward compatibility)
   */
  resetConversation(): void {
    this.clearConversationHistory();
  }


  /**
   * Get current memory context ID
   */
  getCurrentMemoryId(): number | undefined {
    return this.conversationContext.currentMemoryId;
  }

  /**
   * Check if the service is available (simulate health check)
   */
  async checkServiceHealth(): Promise<boolean> {
    try {
      await this.delay(100);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get sequential response based on exact match first, then topic triggers
   * Returns the exact matching response or next response in sequence for the detected topic
   */
  private getSequentialResponse(query: string): VoiceQuery | null {
    const normalizedQuery = query.toLowerCase().trim();
    
    // First, try to find exact match
    const exactMatch = this.demoConversation.find(
      item => item.query.toLowerCase() === normalizedQuery
    );
    
    if (exactMatch) {
      // Update sequence index to the position after this exact match
      const memoryQueries = this.demoConversation.filter(
        item => item.memoryId === exactMatch.memoryId
      );
      const exactMatchIndex = memoryQueries.findIndex(
        item => item.query.toLowerCase() === normalizedQuery
      );
      if (exactMatchIndex !== -1) {
        this.conversationContext.memorySequenceIndex[exactMatch.memoryId] = exactMatchIndex + 1;
      }
      return exactMatch;
    }
    
    // If no exact match, try fuzzy matching with high similarity
    const fuzzyMatch = this.findFuzzyMatch(normalizedQuery);
    if (fuzzyMatch) {
      // Update sequence index to the position after this fuzzy match
      const memoryQueries = this.demoConversation.filter(
        item => item.memoryId === fuzzyMatch.memoryId
      );
      const fuzzyMatchIndex = memoryQueries.findIndex(
        item => item.query === fuzzyMatch.query
      );
      if (fuzzyMatchIndex !== -1) {
        this.conversationContext.memorySequenceIndex[fuzzyMatch.memoryId] = fuzzyMatchIndex + 1;
      }
      return fuzzyMatch;
    }
    
    // Define topic triggers and their corresponding memory IDs
    const topicTriggers = {
      techcrunch: 1,
      stripe: 1,
      crypto: 1,
      jennifer: 1,
      coinbase: 1,
      jake: 2,
      birthday: 2,
      coffee: 3,
      meetup: 3,
      mike: 3,
      lisa: 3,
      lunch: 4,
      team: 4,
      italian: 4,
      tom: 4,
      demo: 5,
      project: 5,
      analytics: 5,
      dashboard: 5
    };

    // Find which topic/memory this query relates to
    let targetMemoryId: number | null = null;
    
    for (const [trigger, memoryId] of Object.entries(topicTriggers)) {
      if (normalizedQuery.includes(trigger)) {
        targetMemoryId = memoryId;
        break;
      }
    }

    if (targetMemoryId === null) {
      return null;
    }

    // Get all queries for this memory ID in order
    const memoryQueries = this.demoConversation.filter(
      item => item.memoryId === targetMemoryId
    );

    if (memoryQueries.length === 0) {
      return null;
    }

    // Get current sequence index for this memory ID
    const currentIndex = this.conversationContext.memorySequenceIndex[targetMemoryId] || 0;
    
    // If we've reached the end of the sequence, start over
    const sequenceIndex = currentIndex >= memoryQueries.length ? 0 : currentIndex;
    
    // Update the sequence index for next time
    this.conversationContext.memorySequenceIndex[targetMemoryId] = sequenceIndex + 1;
    
    return memoryQueries[sequenceIndex];
  }

  /**
   * Find fuzzy match for queries that are similar but not exactly the same
   */
  private findFuzzyMatch(query: string): VoiceQuery | null {
    const queryWords = query.split(' ').filter(word => word.length > 2);
    let bestMatch: VoiceQuery | null = null;
    let highestScore = 0;
    const minMatchThreshold = 0.7; // 70% similarity threshold

    for (const item of this.demoConversation) {
      const itemWords = item.query.toLowerCase().split(' ').filter(word => word.length > 2);
      let matchedWords = 0;
      
      for (const queryWord of queryWords) {
        if (itemWords.some(itemWord => 
          itemWord.includes(queryWord) || 
          queryWord.includes(itemWord) ||
          this.calculateSimilarity(queryWord, itemWord) > 0.8
        )) {
          matchedWords++;
        }
      }
      
      const score = matchedWords / Math.max(queryWords.length, itemWords.length);
      
      if (score > highestScore && score >= minMatchThreshold) {
        highestScore = score;
        bestMatch = item;
      }
    }

    return bestMatch;
  }

  /**
   * Calculate string similarity using Levenshtein distance
   */
  private calculateSimilarity(str1: string, str2: string): number {
    const longer = str1.length > str2.length ? str1 : str2;
    const shorter = str1.length > str2.length ? str2 : str1;
    
    if (longer.length === 0) return 1.0;
    
    const distance = this.levenshteinDistance(longer, shorter);
    return (longer.length - distance) / longer.length;
  }

  /**
   * Calculate Levenshtein distance between two strings
   */
  private levenshteinDistance(str1: string, str2: string): number {
    const matrix = Array(str2.length + 1).fill(null).map(() => Array(str1.length + 1).fill(null));
    
    for (let i = 0; i <= str1.length; i++) matrix[0][i] = i;
    for (let j = 0; j <= str2.length; j++) matrix[j][0] = j;
    
    for (let j = 1; j <= str2.length; j++) {
      for (let i = 1; i <= str1.length; i++) {
        const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1;
        matrix[j][i] = Math.min(
          matrix[j][i - 1] + 1,
          matrix[j - 1][i] + 1,
          matrix[j - 1][i - 1] + indicator
        );
      }
    }
    
    return matrix[str2.length][str1.length];
  }

  /**
   * Generate fallback response for unmatched queries
   */
  private generateFallbackResponse(query: string): string {
    const fallbackResponses = [
      "I don't have that specific information in my memory right now. Could you provide more context or ask about something else?",
      "I'm not finding that in my records. Can you rephrase your question or ask about a different topic?",
      "That doesn't match anything I remember. Try asking about recent meetings, events, or conversations.",
      "I don't have details about that. Would you like to ask about something else from your recent activities?"
    ];

    return fallbackResponses[Math.floor(Math.random() * fallbackResponses.length)];
  }

  /**
   * Utility method to simulate async delays
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get suggested queries (for UI hints)
   */
  getSuggestedQueries(): string[] {
    return [
      "Tell me about TechCrunch",
      "What about Jake?", 
      "Tell me about the coffee meetup",
      "Remind me about the team lunch",
      "What was the project demo about?"
    ];
  }

  /**
   * Reset sequence for a specific memory ID
   */
  resetMemorySequence(memoryId: number): void {
    this.conversationContext.memorySequenceIndex[memoryId] = 0;
  }

  /**
   * Get current sequence index for a memory ID
   */
  getCurrentSequenceIndex(memoryId: number): number {
    return this.conversationContext.memorySequenceIndex[memoryId] || 0;
  }

  /**
   * Get queries by memory ID (for conversation grouping)
   */
  getQueriesByMemoryId(memoryId: number): VoiceQuery[] {
    return this.demoConversation.filter(item => item.memoryId === memoryId);
  }
}

// Export singleton instance
export const voiceService = new VoiceService();

// Export the class for testing purposes
export default VoiceService;
// BitOptimizerStateManager.js
// This utility handles saving and loading the state of the BitAutoOptimizer

class BitOptimizerStateManager {
  constructor() {
    this.storageKey = 'bitAutoOptimizerState';
  }

  // Save the current state of the optimizer
  saveState(state) {
    try {
      // Create a state object with only the essential data
      const stateToSave = {
        currentCode: state.currentCode || '',
        messages: state.messages || [],
        lastUpdated: new Date().toISOString(),
        modelInfo: state.modelInfo || null,
      };
      
      // Use sessionStorage to persist data within the browser session
      // This keeps data when navigating between tabs but clears when browser is closed
      sessionStorage.setItem(this.storageKey, JSON.stringify(stateToSave));
      console.log('BitAutoOptimizer state saved');
      return true;
    } catch (error) {
      console.error('Error saving BitAutoOptimizer state:', error);
      return false;
    }
  }

  // Load the saved state
  loadState() {
    try {
      const savedState = sessionStorage.getItem(this.storageKey);
      if (!savedState) {
        return null;
      }
      
      const parsedState = JSON.parse(savedState);
      
      // Restore Date objects for message timestamps
      if (parsedState.messages && Array.isArray(parsedState.messages)) {
        parsedState.messages = parsedState.messages.map(message => {
          if (message.timestamp) {
            // Convert the string timestamp back to a Date object
            message.timestamp = new Date(message.timestamp);
          }
          return message;
        });
      }
      
      console.log('BitAutoOptimizer state loaded');
      return parsedState;
    } catch (error) {
      console.error('Error loading BitAutoOptimizer state:', error);
      return null;
    }
  }

  // Clear the saved state (used when resetting the optimizer)
  clearState() {
    try {
      sessionStorage.removeItem(this.storageKey);
      console.log('BitAutoOptimizer state cleared');
      return true;
    } catch (error) {
      console.error('Error clearing BitAutoOptimizer state:', error);
      return false;
    }
  }
}

export default new BitOptimizerStateManager();
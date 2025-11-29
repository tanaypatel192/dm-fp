/**
 * Service Worker Configuration
 *
 * Offline support and caching strategies using Workbox
 */

import { Workbox } from 'workbox-window';

let wb: Workbox | undefined;

/**
 * Register service worker
 *
 * Call this once at app startup
 */
export const registerServiceWorker = () => {
  // Only register in production
  if (import.meta.env.MODE !== 'production') {
    console.log('Service worker registration skipped in development');
    return;
  }

  // Check if service workers are supported
  if (!('serviceWorker' in navigator)) {
    console.warn('Service workers not supported in this browser');
    return;
  }

  wb = new Workbox('/service-worker.js');

  // Add event listeners for service worker lifecycle

  // Service worker installed (first time)
  wb.addEventListener('installed', (event) => {
    if (event.isUpdate) {
      console.log('Service worker updated');
      // Optionally show a prompt to reload
      showUpdatePrompt();
    } else {
      console.log('Service worker installed for the first time');
      showOfflineReadyMessage();
    }
  });

  // Service worker activated
  wb.addEventListener('activated', (event) => {
    if (!event.isUpdate) {
      console.log('Service worker activated');
    }
  });

  // Service worker controlling the page
  wb.addEventListener('controlling', () => {
    console.log('Service worker is controlling the page');
    // Reload the page to ensure all assets are from the new service worker
    window.location.reload();
  });

  // Service worker waiting to activate (new version available)
  wb.addEventListener('waiting', () => {
    console.log('New service worker is waiting to activate');
    showUpdatePrompt();
  });

  // Service worker external waiting (opened in another tab)
  wb.addEventListener('externalwaiting', () => {
    console.log('Service worker waiting in another tab');
  });

  // Handle message from service worker
  wb.addEventListener('message', (event) => {
    if (event.data.type === 'CACHE_UPDATED') {
      const { updatedURL } = event.data.payload;
      console.log(`Cache updated for: ${updatedURL}`);
    }
  });

  // Register the service worker
  wb.register()
    .then(() => {
      console.log('✓ Service worker registered successfully');
    })
    .catch((error) => {
      console.error('Service worker registration failed:', error);
    });
};

/**
 * Unregister service worker
 *
 * Use this to remove service worker (cleanup)
 */
export const unregisterServiceWorker = async () => {
  if (!('serviceWorker' in navigator)) {
    return;
  }

  try {
    const registrations = await navigator.serviceWorker.getRegistrations();
    for (const registration of registrations) {
      await registration.unregister();
    }
    console.log('Service worker unregistered');
  } catch (error) {
    console.error('Error unregistering service worker:', error);
  }
};

/**
 * Update service worker
 *
 * Force update to new version
 */
export const updateServiceWorker = async () => {
  if (!wb) {
    console.warn('Service worker not registered');
    return;
  }

  // Tell waiting service worker to activate
  wb.messageSkipWaiting();
};

/**
 * Check for updates
 *
 * Manually check if a new service worker version is available
 */
export const checkForUpdates = async () => {
  if (!wb) {
    console.warn('Service worker not registered');
    return;
  }

  try {
    await wb.update();
    console.log('Checked for service worker updates');
  } catch (error) {
    console.error('Error checking for updates:', error);
  }
};

/**
 * Show update prompt to user
 *
 * Notify user that a new version is available
 */
const showUpdatePrompt = () => {
  // Create a custom event that components can listen to
  const event = new CustomEvent('sw-update-available', {
    detail: {
      update: updateServiceWorker,
    },
  });
  window.dispatchEvent(event);

  // You can also use a toast notification here
  console.log('New version available! Please refresh to update.');
};

/**
 * Show offline ready message
 *
 * Notify user that app is ready for offline use
 */
const showOfflineReadyMessage = () => {
  // Create a custom event
  const event = new CustomEvent('sw-offline-ready');
  window.dispatchEvent(event);

  console.log('App is ready for offline use!');
};

/**
 * Check if app is running offline
 */
export const isOffline = (): boolean => {
  return !navigator.onLine;
};

/**
 * Add offline/online event listeners
 */
export const addNetworkStatusListeners = (
  onOnline: () => void,
  onOffline: () => void
) => {
  window.addEventListener('online', onOnline);
  window.addEventListener('offline', onOffline);

  // Return cleanup function
  return () => {
    window.removeEventListener('online', onOnline);
    window.removeEventListener('offline', onOffline);
  };
};

/**
 * Network status hook
 *
 * React hook to track online/offline status
 */
export const useNetworkStatus = () => {
  const [isOnline, setIsOnline] = React.useState(navigator.onLine);

  React.useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    const cleanup = addNetworkStatusListeners(handleOnline, handleOffline);

    return cleanup;
  }, []);

  return isOnline;
};

/**
 * Service worker update hook
 *
 * React hook to handle service worker updates
 */
export const useServiceWorkerUpdate = () => {
  const [updateAvailable, setUpdateAvailable] = React.useState(false);
  const [updateCallback, setUpdateCallback] = React.useState<(() => void) | null>(null);

  React.useEffect(() => {
    const handleUpdateAvailable = (event: Event) => {
      const customEvent = event as CustomEvent;
      setUpdateAvailable(true);
      setUpdateCallback(() => customEvent.detail.update);
    };

    window.addEventListener('sw-update-available', handleUpdateAvailable);

    return () => {
      window.removeEventListener('sw-update-available', handleUpdateAvailable);
    };
  }, []);

  const applyUpdate = () => {
    if (updateCallback) {
      updateCallback();
    }
  };

  const dismissUpdate = () => {
    setUpdateAvailable(false);
  };

  return {
    updateAvailable,
    applyUpdate,
    dismissUpdate,
  };
};

/**
 * Cache management utilities
 */
export const cacheUtils = {
  /**
   * Clear all caches
   */
  clearAll: async () => {
    if (!('caches' in window)) return;

    try {
      const cacheNames = await caches.keys();
      await Promise.all(cacheNames.map((name) => caches.delete(name)));
      console.log('All caches cleared');
    } catch (error) {
      console.error('Error clearing caches:', error);
    }
  },

  /**
   * Get cache size
   */
  getSize: async (): Promise<number> => {
    if (!('caches' in window)) return 0;

    try {
      const cacheNames = await caches.keys();
      let totalSize = 0;

      for (const name of cacheNames) {
        const cache = await caches.open(name);
        const keys = await cache.keys();

        for (const request of keys) {
          const response = await cache.match(request);
          if (response) {
            const blob = await response.blob();
            totalSize += blob.size;
          }
        }
      }

      return totalSize;
    } catch (error) {
      console.error('Error calculating cache size:', error);
      return 0;
    }
  },

  /**
   * Get cache statistics
   */
  getStats: async () => {
    if (!('caches' in window)) {
      return {
        cacheCount: 0,
        totalSize: 0,
        caches: [],
      };
    }

    try {
      const cacheNames = await caches.keys();
      const cacheStats = [];

      for (const name of cacheNames) {
        const cache = await caches.open(name);
        const keys = await cache.keys();
        let cacheSize = 0;

        for (const request of keys) {
          const response = await cache.match(request);
          if (response) {
            const blob = await response.blob();
            cacheSize += blob.size;
          }
        }

        cacheStats.push({
          name,
          entries: keys.length,
          size: cacheSize,
        });
      }

      const totalSize = cacheStats.reduce((sum, cache) => sum + cache.size, 0);

      return {
        cacheCount: cacheNames.length,
        totalSize,
        caches: cacheStats,
      };
    } catch (error) {
      console.error('Error getting cache stats:', error);
      return {
        cacheCount: 0,
        totalSize: 0,
        caches: [],
      };
    }
  },
};

/**
 * Prefetch important resources
 *
 * Use this to warm up the cache with critical resources
 */
export const prefetchResources = async (urls: string[]) => {
  if (!wb) {
    console.warn('Service worker not registered');
    return;
  }

  try {
    await Promise.all(
      urls.map(async (url) => {
        try {
          await fetch(url);
        } catch (error) {
          console.warn(`Failed to prefetch: ${url}`);
        }
      })
    );
    console.log('Resources prefetched successfully');
  } catch (error) {
    console.error('Error prefetching resources:', error);
  }
};

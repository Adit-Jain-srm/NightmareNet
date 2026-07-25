import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useSounds } from "@/lib/sounds";

// Mock AudioContext since jsdom doesn't have Web Audio API
const mockOscillator = {
  type: "sine",
  frequency: { setValueAtTime: vi.fn() },
  connect: vi.fn().mockReturnThis(),
  start: vi.fn(),
  stop: vi.fn(),
};

const mockGain = {
  gain: {
    setValueAtTime: vi.fn(),
    exponentialRampToValueAtTime: vi.fn(),
    linearRampToValueAtTime: vi.fn(),
  },
  connect: vi.fn().mockReturnThis(),
};

const mockAudioContext = {
  state: "running",
  currentTime: 0,
  sampleRate: 44100,
  resume: vi.fn(),
  createOscillator: vi.fn(() => mockOscillator),
  createGain: vi.fn(() => mockGain),
  createBuffer: vi.fn(() => ({
    getChannelData: vi.fn(() => new Float32Array(2646)),
  })),
  createBufferSource: vi.fn(() => ({
    buffer: null,
    connect: vi.fn().mockReturnThis(),
    start: vi.fn(),
    stop: vi.fn(),
  })),
  createBiquadFilter: vi.fn(() => ({
    type: "bandpass",
    frequency: { setValueAtTime: vi.fn() },
    Q: { setValueAtTime: vi.fn() },
    connect: vi.fn().mockReturnThis(),
  })),
  destination: {},
};

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: vi.fn((key: string) => { delete store[key]; }),
    clear: vi.fn(() => { store = {}; }),
  };
})();

describe("useSounds hook", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorageMock.clear();

    // Reset stubs before each test to guarantee isolated states
    vi.stubGlobal("localStorage", localStorageMock);
    vi.stubGlobal("matchMedia", vi.fn((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns play functions for every supported sound", () => {
    vi.stubGlobal("AudioContext", function() { return mockAudioContext; });

    const { result } = renderHook(() => useSounds());

    expect(typeof result.current.playClick).toBe("function");
    expect(typeof result.current.playSuccess).toBe("function");
    expect(typeof result.current.playError).toBe("function");
    expect(typeof result.current.playTransition).toBe("function");
    expect(typeof result.current.playNotification).toBe("function");
    expect(typeof result.current.toggle).toBe("function");
    expect(result.current.enabled).toBeDefined();
  });

  it("respects the user's mute preference", () => {
    vi.stubGlobal("AudioContext", function() { return mockAudioContext; });

    // Simulate user previously disabling sounds via localStorage
    localStorageMock.getItem.mockReturnValueOnce("false");

    const { result } = renderHook(() => useSounds());

    // Ensure hook initialized in a muted state
    expect(result.current.enabled).toBe(false);

    act(() => {
      result.current.playClick();
    });

    // AudioContext should not be interacted with because sounds are disabled
    expect(mockAudioContext.createOscillator).not.toHaveBeenCalled();

    // User toggles sounds back on
    act(() => {
      result.current.toggle();
    });

    expect(result.current.enabled).toBe(true);

    act(() => {
      result.current.playClick();
    });

    // Now AudioContext should generate the sound
    expect(mockAudioContext.createOscillator).toHaveBeenCalled();
  });

  it("does not throw when Audio API is unavailable (SSR-safe)", () => {
    // Deliberately make AudioContext unavailable to simulate SSR or unsupported browser
    vi.stubGlobal("AudioContext", undefined);

    const { result } = renderHook(() => useSounds());

    expect(() => {
      act(() => {
        result.current.playClick();
        result.current.playSuccess();
        result.current.playError();
        result.current.playTransition();
        result.current.playNotification();
      });
    }).not.toThrow();
  });
});

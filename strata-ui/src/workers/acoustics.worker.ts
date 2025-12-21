/**
 * Web Worker for acoustic analysis computation.
 *
 * Offloads computationally intensive impulse response and acoustic metrics
 * analysis to a background thread to keep the main thread responsive.
 */

import {
  analyzeTransferFunction,
  type ImpulseResponseResult,
  type EnergyDecayResult,
  type AcousticMetrics,
  type WindowType,
} from "@/lib/acoustics";

// =============================================================================
// Message Types
// =============================================================================

export interface AcousticsWorkerRequest {
  type: "analyzeTransferFunction";
  id: number;
  transferReal: Float32Array;
  transferImag: Float32Array;
  sampleRate: number;
  windowType: WindowType;
}

export interface AcousticsWorkerResponse {
  type: "analyzeTransferFunction";
  id: number;
  impulseResponse: ImpulseResponseResult;
  energyDecay: EnergyDecayResult;
  metrics: AcousticMetrics;
}

export interface AcousticsWorkerError {
  type: "error";
  id: number;
  error: string;
}

export type AcousticsWorkerMessage = AcousticsWorkerResponse | AcousticsWorkerError;

// =============================================================================
// Worker Logic
// =============================================================================

self.onmessage = (event: MessageEvent<AcousticsWorkerRequest>) => {
  const request = event.data;

  try {
    switch (request.type) {
      case "analyzeTransferFunction": {
        const result = analyzeTransferFunction(
          request.transferReal,
          request.transferImag,
          request.sampleRate,
          request.windowType
        );

        const response: AcousticsWorkerResponse = {
          type: "analyzeTransferFunction",
          id: request.id,
          impulseResponse: result.impulseResponse,
          energyDecay: result.energyDecay,
          metrics: result.metrics,
        };

        // Transfer ownership of buffers for zero-copy
        self.postMessage(response, {
          transfer: [
            result.impulseResponse.impulseResponse.buffer,
            result.impulseResponse.timeAxis.buffer,
            result.energyDecay.decayCurve.buffer,
            result.energyDecay.timeAxis.buffer,
          ],
        });
        break;
      }
    }
  } catch (error) {
    const errorResponse: AcousticsWorkerError = {
      type: "error",
      id: request.id,
      error: error instanceof Error ? error.message : "Unknown error",
    };
    self.postMessage(errorResponse);
  }
};

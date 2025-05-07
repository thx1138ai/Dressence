declare module 'react-native-canvas' {
    import React from 'react';
  
    export interface CanvasRenderingContext2D {
      drawImage(image: CanvasImage, x: number, y: number, width?: number, height?: number): void;
      getImageData(x: number, y: number, width: number, height: number): Promise<{ data: Uint8ClampedArray }>;
      // Add other methods as needed
    }
  
    export class Canvas extends React.Component<any> {
      width: number;
      height: number;
      getContext(contextType: string): CanvasRenderingContext2D | null;
      addEventListener(event: string, callback: Function): void;
      removeEventListener(event: string, callback: Function): void;
    }
  
    export class CanvasImage {
      src: string;
      addEventListener(event: string, callback: Function): void;
      removeEventListener(event: string, callback: Function): void;
    }
  }
  
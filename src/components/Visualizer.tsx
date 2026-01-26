import React, { useRef, useEffect } from 'react';

export const Visualizer: React.FC<{ isActive: boolean; analyser: AnalyserNode | null }> = ({ isActive, analyser }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !analyser || !isActive) return;
    const ctx = canvasRef.current.getContext('2d')!;
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    let animationId: number;

    const draw = () => {
      animationId = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);
      ctx.clearRect(0, 0, 600, 128);
      let x = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const barHeight = (dataArray[i] / 255) * 128;
        ctx.fillStyle = '#3b82f6'; // Tailwind blue-500
        ctx.fillRect(x, 128 - barHeight, 3, barHeight);
        x += 4;
      }
    };
    draw();
    return () => cancelAnimationFrame(animationId);
  }, [isActive, analyser]);

  return <canvas ref={canvasRef} className="w-full h-32 rounded-xl bg-slate-900/50" width={600} height={128} />;
};
import * as THREE from 'three';
import { dyno } from '@sparkjsdev/spark';

export function makeSemanticModifier(
  textureUniform: dyno.DynoUsampler2D<string, THREE.DataTexture>, 
  displayModeUniform: dyno.DynoInt<string>
) {
  return dyno.dynoBlock(
    { gsplat: dyno.Gsplat }, 
    { gsplat: dyno.Gsplat }, 
    ({ gsplat }) => {
      if (!gsplat) {
        throw new Error("No gsplat input");
      }

      // Split splat to get index and current rgb/opacity
      const splatSplit = dyno.splitGsplat(gsplat).outputs;
      const index = splatSplit.index;
      const rgb = splatSplit.rgb;
      const opacity = splatSplit.opacity;

      // Get texture size
      const texSizeVal = dyno.textureSize(textureUniform); // ivec2
      const { x: width } = dyno.split(texSizeVal).outputs;

      // Calculate x and y coordinates for texelFetch
      const x = dyno.imod(index, width);
      const y = dyno.div(index, width);
      const coord = dyno.combine({ vectorType: "ivec2", x, y });

      // Read the value from the DataTexture
      const valueVec4 = dyno.texelFetch(textureUniform, coord);
      const { r: isolatedValueInt } = dyno.split(valueVec4).outputs; // uint

      // Convert uint to bool
      const isIsolated = dyno.equal(isolatedValueInt, dyno.dynoConst("uint", 1));

      // Modes: 0 = Default, 1 = Highlight, 2 = Isolate
      const mode1 = dyno.equal(displayModeUniform, dyno.dynoConst("int", 1));
      const mode2 = dyno.equal(displayModeUniform, dyno.dynoConst("int", 2));

      // Mode 1: Highlight the object (mix base color with neon green)
      const neonGreen = dyno.combine({ 
        vectorType: "vec3", 
        r: dyno.dynoConst("float", 0.0), 
        g: dyno.dynoConst("float", 1.0), 
        b: dyno.dynoConst("float", 0.0) 
      });
      const highlightedRgb = dyno.mix(rgb, neonGreen, dyno.dynoConst("float", 0.5));
      
      // Mode 2: Isolate the object (hide background)
      const hiddenOpacity = dyno.dynoConst("float", 0.0);

      // Apply logic
      const finalRgb = dyno.select(
        mode1, 
        dyno.select(isIsolated, highlightedRgb, rgb), 
        rgb
      );

      const finalOpacity = dyno.select(
        mode2,
        dyno.select(isIsolated, opacity, hiddenOpacity),
        opacity
      );

      // Recombine and return
      gsplat = dyno.combineGsplat({ gsplat, rgb: finalRgb, opacity: finalOpacity });
      return { gsplat };
    }
  );
}

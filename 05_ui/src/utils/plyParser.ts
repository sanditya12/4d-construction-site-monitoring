export function extractIsolatedObject(fileBytes: ArrayBuffer): Uint8Array | null {
  const dataView = new DataView(fileBytes);
  const textDecoder = new TextDecoder('utf-8');
  
  // Find the end of the header
  let headerEndIndex = 0;
  for (let i = 0; i < 10000; i++) { // Header should be within the first 10k bytes
    if (
      dataView.getUint8(i) === 10 && // \n
      dataView.getUint8(i - 1) === 114 && // r
      dataView.getUint8(i - 2) === 101 && // e
      dataView.getUint8(i - 3) === 100 && // d
      dataView.getUint8(i - 4) === 97 && // a
      dataView.getUint8(i - 5) === 101 && // e
      dataView.getUint8(i - 6) === 104 && // h
      dataView.getUint8(i - 7) === 95 && // _
      dataView.getUint8(i - 8) === 100 && // d
      dataView.getUint8(i - 9) === 110 && // n
      dataView.getUint8(i - 10) === 101 // e
    ) {
      headerEndIndex = i + 1;
      break;
    }
  }

  if (headerEndIndex === 0) {
    console.warn("Could not find end_header in PLY file.");
    return null;
  }

  const headerStr = textDecoder.decode(new Uint8Array(fileBytes, 0, headerEndIndex));
  const lines = headerStr.split('\n').map(l => l.trim()).filter(l => l.length > 0);

  if (!lines.includes('format binary_little_endian 1.0')) {
    console.warn("PLY is not binary_little_endian. Semantic parsing aborted.");
    return null;
  }

  let inVertexElement = false;
  let vertexCount = 0;
  let currentOffset = 0;
  let isolatedObjectOffset = -1;
  let vertexStride = 0;

  for (const line of lines) {
    const parts = line.split(' ');
    if (parts[0] === 'element') {
      if (parts[1] === 'vertex') {
        inVertexElement = true;
        vertexCount = parseInt(parts[2], 10);
      } else {
        inVertexElement = false;
      }
    } else if (parts[0] === 'property' && inVertexElement) {
      const type = parts[1];
      const name = parts[2];
      
      let size = 0;
      switch (type) {
        case 'char': case 'uchar': case 'int8': case 'uint8': size = 1; break;
        case 'short': case 'ushort': case 'int16': case 'uint16': size = 2; break;
        case 'int': case 'uint': case 'int32': case 'uint32': case 'float': case 'float32': size = 4; break;
        case 'double': case 'float64': size = 8; break;
        default: size = 4; // fallback
      }

      if (name === 'isolated_object') {
        isolatedObjectOffset = currentOffset;
      }
      currentOffset += size;
    }
  }

  if (isolatedObjectOffset === -1) {
    console.warn("isolated_object property not found in PLY header.");
    return null;
  }

  vertexStride = currentOffset;
  const result = new Uint8Array(vertexCount);
  
  // Extract the data
  for (let i = 0; i < vertexCount; i++) {
    const byteIndex = headerEndIndex + i * vertexStride + isolatedObjectOffset;
    if (byteIndex < fileBytes.byteLength) {
      result[i] = dataView.getUint8(byteIndex);
    }
  }

  console.log(`Extracted isolated_object data for ${vertexCount} splats.`);
  return result;
}

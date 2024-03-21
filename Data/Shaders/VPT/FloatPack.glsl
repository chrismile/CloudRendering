// Converts normalized float in range [0,1] to 32-bit uint.
uint convertNormalizedFloatToUint32(float valueFloat) {
    return uint(round(clamp(valueFloat, 0.0, 1.0) * 4294967295.0));
}

// Decompression equivalent to function above.
float convertUint32ToNormalizedFloat(uint valueUint) {
    return float(valueUint) / 4294967295.0;
}

#ifndef __UTILS_H__
#define __UTILS_H__

inline int SMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
  typedef struct {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
    int Cores;
  } SM2Cores;

  SM2Cores coresPerSM[] = {
    {0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
    {0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
    {0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
    {0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
    {0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
    {0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
    {0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
    {0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
    {0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
    {0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
    {0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
    {0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
    {  -1, -1 }
  };

  int idx = 0;
  for(idx = 0; coresPerSM[idx].SM != -1; idx++) {
    if (coresPerSM[idx].SM == ((major << 4) + minor)) {
      return coresPerSM[idx].Cores;
    }
  }
  return coresPerSM[idx - 1].Cores;
}

int BestDevice();

#endif // __UTILS_H__

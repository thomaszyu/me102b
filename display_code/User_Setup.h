// ----- TFT_eSPI User_Setup for 4.0" ESP32-32E Display (LCDwiki) -----
// Copy this file into your TFT_eSPI library folder, replacing the default User_Setup.h
// Library location (Arduino IDE):
//   macOS:  ~/Documents/Arduino/libraries/TFT_eSPI/
//   Linux:  ~/Arduino/libraries/TFT_eSPI/
//   Windows: Documents\Arduino\libraries\TFT_eSPI\

#define ST7796_DRIVER

#define TFT_WIDTH  320
#define TFT_HEIGHT 480

// --- SPI pin assignments (HSPI) ---
#define TFT_MISO  12
#define TFT_MOSI  13
#define TFT_SCLK  14
#define TFT_CS    15
#define TFT_DC     2
#define TFT_RST   -1   // Connected to ESP32 EN (hardware reset)

#define USE_HSPI_PORT

// --- Backlight ---
#define TFT_BL    27
#define TFT_BACKLIGHT_ON HIGH

// --- Touch (XPT2046, shares SPI bus) ---
#define TOUCH_CS  33

// --- SPI speeds ---
#define SPI_FREQUENCY       40000000
#define SPI_READ_FREQ       20000000
#define SPI_TOUCH_FREQ       2500000

// --- Fonts ---
#define LOAD_GLCD
#define LOAD_FONT2
#define LOAD_FONT4
#define LOAD_FONT6
#define LOAD_FONT7
#define LOAD_FONT8
#define LOAD_GFXFF
#define SMOOTH_FONT

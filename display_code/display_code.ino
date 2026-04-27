/*
 * Air Hockey Display Controller
 *
 * ESP32-32E + 4.0" ST7796S (LCDwiki board)
 * Menu screen: difficulty selector
 * Game screen: live table view (puck, mallet, strategy) fed from laptop.
 *
 * Setup:
 *   1. Install TFT_eSPI via Arduino Library Manager
 *   2. Copy User_Setup.h from this folder into the TFT_eSPI library folder,
 *      replacing the default one.
 *   3. In Arduino IDE: Tools > Board > ESP32 Dev Module
 *   4. Upload, then run on the laptop:
 *        python laptop_listener.py --serial /dev/cu.usbserial-XXXX
 *
 * Touch calibration:
 *   If touch coordinates are off, uncomment CALIBRATE_TOUCH below,
 *   upload, tap the corners, then copy the printed values into touchCalData.
 *
 */

#include <TFT_eSPI.h>
#include <SPI.h>

// Uncomment to run touch calibration on boot
// #define CALIBRATE_TOUCH

// ===================== CONFIG =====================
const uint32_t SERIAL_BAUD = 115200;
// ==================================================

// ===================== DISPLAY ====================
#define LCD_BL 27 // Backlight pin

TFT_eSPI tft = TFT_eSPI();

// Touch calibration data (run calibration if touch is inaccurate)
uint16_t touchCalData[5] = {300, 3600, 300, 3600, 3};

// ===================== COLORS =====================
#define COL_BG 0x1082 // dark charcoal
#define COL_TITLE TFT_WHITE
#define COL_SUBTITLE 0xC618 // light grey
#define COL_EASY 0x2DC4     // green
#define COL_MEDIUM 0xFDA0   // orange
#define COL_HARD 0xF800     // red
#define COL_START 0x34DF    // blue
#define COL_SELECT 0x07FF   // cyan highlight
#define COL_BTN_TEXT TFT_WHITE

// Game view colors
#define COL_TABLE 0x6B6D // grey border
#define COL_PUCK_BND TFT_WHITE
#define COL_PUCK 0x07E0   // green
#define COL_MALLET 0xFDA0 // orange
#define COL_DEFEND 0x04DF // orange-ish (TFT_eSPI: R5G6B5)
#define COL_OUR_GOAL TFT_RED
#define COL_THEIR_GOAL 0xF81F // magenta
#define COL_CENTER 0x4208     // dim grey
#define COL_WORKSPACE 0xFFE0  // yellow
#define COL_STRATEGY TFT_WHITE

// ===================== SCREENS ====================
enum Screen : uint8_t
{
    SCREEN_MENU,
    SCREEN_GAME
};
Screen currentScreen = SCREEN_MENU;

// ===================== STATE ======================
enum Difficulty : uint8_t
{
    DIFF_NONE = 0,
    DIFF_EASY,
    DIFF_MEDIUM,
    DIFF_HARD
};
Difficulty selectedDifficulty = DIFF_NONE;
bool lastTouchState = false;

// ===================== UI LAYOUT ==================
// Landscape: 480 x 320
#define SCREEN_W 480
#define SCREEN_H 320

struct Button
{
    int16_t x, y, w, h;
    const char *label;
    uint16_t color;
};

#define BTN_W 220
#define BTN_H 48
#define BTN_X ((SCREEN_W - BTN_W) / 2)

Button btnEasy = {BTN_X, 95, BTN_W, BTN_H, "EASY", COL_EASY};
Button btnMedium = {BTN_X, 153, BTN_W, BTN_H, "MEDIUM", COL_MEDIUM};
Button btnHard = {BTN_X, 211, BTN_W, BTN_H, "HARD", COL_HARD};
Button btnStart = {BTN_X, 272, BTN_W, 40, "START GAME", COL_START};

// ===================== TABLE CONSTANTS (mm) =======
// Matches air_hockey_player.py
#define TBL_X_MIN -273.0f
#define TBL_X_MAX 273.0f
#define TBL_Y_MIN -240.0f
#define TBL_Y_MAX 240.0f
#define TBL_DEFEND_X (TBL_X_MIN + 120.0f) // -153
#define TBL_ATTACK_ZONE 120.0f
#define TBL_GOAL_HALF 80.0f

// Display area for the table (shrunk to fit buttons below)
#define TABLE_MARGIN 15
#define TABLE_TOP 25    // room for score bar
#define TABLE_BOTTOM 50 // room for pause/stop buttons
#define TABLE_PX_X TABLE_MARGIN
#define TABLE_PX_Y TABLE_TOP
#define TABLE_PX_W (SCREEN_W - 2 * TABLE_MARGIN)
#define TABLE_PX_H (SCREEN_H - TABLE_TOP - TABLE_BOTTOM)

// Pause/Stop button layout
#define GBTN_W 100
#define GBTN_H 35
#define GBTN_Y (SCREEN_H - GBTN_H - 5)
Button btnPause = {SCREEN_W / 2 - GBTN_W - 10, GBTN_Y, GBTN_W, GBTN_H, "PAUSE", TFT_YELLOW};
Button btnStop = {SCREEN_W / 2 + 10, GBTN_Y, GBTN_W, GBTN_H, "STOP", TFT_RED};

// ===================== GAME STATE =================
// Received from laptop via UART
float puckX = 0, puckY = 0, puckVX = 0, puckVY = 0;
bool puckValid = false;
float malletX = 0, malletY = 0;
bool malletValid = false;
char strategy[16] = "IDLE";
int scoreUs = 0, scoreThem = 0;

// Previous positions for efficient redraw (erase old, draw new)
int16_t prevPuckPX = -1, prevPuckPY = -1;
int16_t prevMalletPX = -1, prevMalletPY = -1;

// Attack trajectory (from laptop)
#define MAX_TRAJ_PTS 20
float trajX[MAX_TRAJ_PTS], trajY[MAX_TRAJ_PTS];
int trajLen = 0;
int trajContactIdx = -1; // index of the contact point, -1 if none
float trajPrevPxX[MAX_TRAJ_PTS], trajPrevPxY[MAX_TRAJ_PTS];
int prevTrajLen = 0;

// Goal banner — when the laptop fires a {"type":"goal", ...} message we
// pop a 3-second banner across the table area announcing who scored.
#define GOAL_BANNER_MS 3000
unsigned long goalBannerEndMs = 0;
char goalBannerText[24] = "";
uint16_t goalBannerColor = TFT_WHITE;

// ===================== COORDINATE MAPPING =========

int16_t mmToPxX(float x_mm)
{
    float pad = 20.0f;
    return TABLE_PX_X + (int16_t)((x_mm - (TBL_X_MIN - pad)) / ((TBL_X_MAX + pad) - (TBL_X_MIN - pad)) * TABLE_PX_W);
}

int16_t mmToPxY(float y_mm)
{
    float pad = 20.0f;
    // Y is flipped (positive Y = up in mm, but down in pixels)
    return TABLE_PX_Y + (int16_t)(((TBL_Y_MAX + pad) - y_mm) / ((TBL_Y_MAX + pad) - (TBL_Y_MIN - pad)) * TABLE_PX_H);
}

// ===================== MENU FUNCTIONS =============

void drawButton(Button &btn, bool selected)
{
    if (selected)
    {
        tft.drawRoundRect(btn.x - 3, btn.y - 3, btn.w + 6, btn.h + 6, 11, COL_SELECT);
        tft.drawRoundRect(btn.x - 2, btn.y - 2, btn.w + 4, btn.h + 4, 10, COL_SELECT);
    }
    else
    {
        tft.drawRoundRect(btn.x - 3, btn.y - 3, btn.w + 6, btn.h + 6, 11, COL_BG);
        tft.drawRoundRect(btn.x - 2, btn.y - 2, btn.w + 4, btn.h + 4, 10, COL_BG);
    }
    tft.fillRoundRect(btn.x, btn.y, btn.w, btn.h, 8, btn.color);
    tft.setTextColor(COL_BTN_TEXT);
    tft.setTextDatum(MC_DATUM);
    tft.drawString(btn.label, btn.x + btn.w / 2, btn.y + btn.h / 2, 4);
}

void drawMenuUI()
{
    tft.fillScreen(COL_BG);
    tft.setTextColor(COL_TITLE);
    tft.setTextDatum(TC_DATUM);
    tft.drawString("AIR HOCKEY", SCREEN_W / 2, 30, 4);
    tft.setTextColor(COL_SUBTITLE);
    tft.drawString("Select Difficulty", SCREEN_W / 2, 65, 2);
    drawButton(btnEasy, selectedDifficulty == DIFF_EASY);
    drawButton(btnMedium, selectedDifficulty == DIFF_MEDIUM);
    drawButton(btnHard, selectedDifficulty == DIFF_HARD);
    drawButton(btnStart, false);
}

bool getTouch(uint16_t &tx, uint16_t &ty)
{
    uint16_t rawx, rawy;
    bool pressed = tft.getTouch(&rawx, &rawy, 40);
    if (pressed && rawx < SCREEN_W && rawy < SCREEN_H)
    {
        tx = SCREEN_W - 1 - rawx; // mirror X to match rotation 3
        ty = rawy;
        return true;
    }
    return false;
}

bool touchInside(Button &btn, uint16_t tx, uint16_t ty)
{
    return tx >= btn.x && tx <= (btn.x + btn.w) &&
           ty >= btn.y && ty <= (btn.y + btn.h);
}

// ===================== GAME VIEW ==================

void drawTableStatic()
{
    // Called once when entering game screen — draws all static elements

    tft.fillScreen(TFT_BLACK);

    // Table border (outer wall)
    int16_t x1 = mmToPxX(TBL_X_MIN);
    int16_t y1 = mmToPxY(TBL_Y_MAX);
    int16_t x2 = mmToPxX(TBL_X_MAX);
    int16_t y2 = mmToPxY(TBL_Y_MIN);
    tft.drawRect(x1, y1, x2 - x1, y2 - y1, COL_TABLE);

    // Puck boundary (white, slightly inset)
    float pr = 25.0f; // PUCK_RADIUS
    int16_t px1 = mmToPxX(TBL_X_MIN + pr);
    int16_t py1 = mmToPxY(TBL_Y_MAX - pr);
    int16_t px2 = mmToPxX(TBL_X_MAX - pr);
    int16_t py2 = mmToPxY(TBL_Y_MIN + pr);
    tft.drawRect(px1, py1, px2 - px1, py2 - py1, COL_PUCK_BND);

    // Center line
    int16_t cx = mmToPxX(0);
    tft.drawLine(cx, y1, cx, y2, COL_CENTER);

    // Defense line
    int16_t dx = mmToPxX(TBL_DEFEND_X);
    tft.drawLine(dx, y1, dx, y2, COL_DEFEND);
    tft.setTextColor(COL_DEFEND, TFT_BLACK);
    tft.setTextDatum(BC_DATUM);
    tft.drawString("DEF", dx, y1 - 2, 1);

    // Attack zone line
    int16_t ax = mmToPxX(TBL_ATTACK_ZONE);
    tft.drawLine(ax, y1, ax, y2, COL_CENTER);
    tft.setTextColor(COL_CENTER, TFT_BLACK);
    tft.setTextDatum(BC_DATUM);
    tft.drawString("ATK", ax, y1 - 2, 1);

    // Our goal (left, red)
    int16_t gy1 = mmToPxY(TBL_GOAL_HALF);
    int16_t gy2 = mmToPxY(-TBL_GOAL_HALF);
    tft.drawLine(x1, gy1, x1, gy2, COL_OUR_GOAL);
    tft.drawLine(x1 - 1, gy1, x1 - 1, gy2, COL_OUR_GOAL);

    // Their goal (right, magenta)
    tft.drawLine(x2, gy1, x2, gy2, COL_THEIR_GOAL);
    tft.drawLine(x2 + 1, gy1, x2 + 1, gy2, COL_THEIR_GOAL);

    // Score header
    drawScoreBar();

    // Pause / Stop buttons
    drawButton(btnPause, false);
    drawButton(btnStop, false);

    // "Waiting for data..." initially
    tft.setTextColor(COL_SUBTITLE, TFT_BLACK);
    tft.setTextDatum(MC_DATUM);
    tft.drawString("Waiting for data...", SCREEN_W / 2,
                   TABLE_PX_Y + TABLE_PX_H / 2, 2);
}

void drawScoreBar()
{
    // Top bar with score (compact)
    tft.fillRect(0, 0, SCREEN_W, TABLE_TOP - 2, TFT_BLACK);
    tft.setTextDatum(TL_DATUM);

    // Our score (left)
    tft.setTextColor(COL_OUR_GOAL, TFT_BLACK);
    char buf[32];
    snprintf(buf, sizeof(buf), "US:%d", scoreUs);
    tft.drawString(buf, 5, 4, 2);

    // Their score (right)
    tft.setTextColor(COL_THEIR_GOAL, TFT_BLACK);
    snprintf(buf, sizeof(buf), "THEM:%d", scoreThem);
    tft.setTextDatum(TR_DATUM);
    tft.drawString(buf, SCREEN_W - 5, 4, 2);

    // Strategy (center)
    tft.setTextDatum(TC_DATUM);
    uint16_t stratColor = COL_STRATEGY;
    if (strcmp(strategy, "DEFEND") == 0)
        stratColor = COL_DEFEND;
    else if (strcmp(strategy, "STRIKE") == 0)
        stratColor = TFT_CYAN;
    else if (strcmp(strategy, "WINDUP") == 0)
        stratColor = TFT_YELLOW;
    else if (strcmp(strategy, "PAUSED") == 0)
        stratColor = TFT_YELLOW;
    tft.setTextColor(stratColor, TFT_BLACK);
    snprintf(buf, sizeof(buf), " %s ", strategy);
    tft.drawString(buf, SCREEN_W / 2, 4, 2);
}

int prevTrajContactIdx = -1;

void redrawTableLines()
{
    // Repaint static lines that erase passes may have damaged
    int16_t x1 = mmToPxX(TBL_X_MIN);
    int16_t y1 = mmToPxY(TBL_Y_MAX);
    int16_t x2 = mmToPxX(TBL_X_MAX);
    int16_t y2 = mmToPxY(TBL_Y_MIN);
    tft.drawRect(x1, y1, x2 - x1, y2 - y1, COL_TABLE);

    float pr = 25.0f;
    tft.drawRect(mmToPxX(TBL_X_MIN + pr), mmToPxY(TBL_Y_MAX - pr),
                 mmToPxX(TBL_X_MAX - pr) - mmToPxX(TBL_X_MIN + pr),
                 mmToPxY(TBL_Y_MIN + pr) - mmToPxY(TBL_Y_MAX - pr), COL_PUCK_BND);

    tft.drawLine(mmToPxX(0), y1, mmToPxX(0), y2, COL_CENTER);
    tft.drawLine(mmToPxX(TBL_DEFEND_X), y1, mmToPxX(TBL_DEFEND_X), y2, COL_DEFEND);
    tft.drawLine(mmToPxX(TBL_ATTACK_ZONE), y1, mmToPxX(TBL_ATTACK_ZONE), y2, COL_CENTER);

    int16_t gy1 = mmToPxY(TBL_GOAL_HALF);
    int16_t gy2 = mmToPxY(-TBL_GOAL_HALF);
    tft.drawLine(x1, gy1, x1, gy2, COL_OUR_GOAL);
    tft.drawLine(x1 - 1, gy1, x1 - 1, gy2, COL_OUR_GOAL);
    tft.drawLine(x2, gy1, x2, gy2, COL_THEIR_GOAL);
    tft.drawLine(x2 + 1, gy1, x2 + 1, gy2, COL_THEIR_GOAL);
}

void eraseAllDynamic()
{
    // Erase old trajectory
    for (int i = 0; i < prevTrajLen - 1; i++)
    {
        tft.drawLine((int16_t)trajPrevPxX[i], (int16_t)trajPrevPxY[i],
                     (int16_t)trajPrevPxX[i + 1], (int16_t)trajPrevPxY[i + 1], TFT_BLACK);
    }
    if (prevTrajLen > 0 && prevTrajContactIdx >= 0 && prevTrajContactIdx < prevTrajLen)
    {
        tft.drawCircle((int16_t)trajPrevPxX[prevTrajContactIdx],
                       (int16_t)trajPrevPxY[prevTrajContactIdx], 5, TFT_BLACK);
    }

    // Erase old puck
    if (prevPuckPX >= 0)
    {
        tft.fillCircle(prevPuckPX, prevPuckPY, 8, TFT_BLACK);
    }

    // Erase old mallet
    if (prevMalletPX >= 0)
    {
        tft.fillCircle(prevMalletPX, prevMalletPY, 10, TFT_BLACK);
    }

    // Repair any static lines the erases damaged
    redrawTableLines();
}

void drawAllDynamic()
{
    // --- Trajectory ---
    if (trajLen >= 2)
    {
        for (int i = 0; i < trajLen; i++)
        {
            trajPrevPxX[i] = mmToPxX(trajX[i]);
            trajPrevPxY[i] = mmToPxY(trajY[i]);
        }
        for (int i = 0; i < trajLen - 1; i++)
        {
            uint16_t col;
            if (trajContactIdx < 0)
            {
                col = TFT_CYAN;
            }
            else if (i < trajContactIdx)
            {
                col = TFT_CYAN;
            }
            else if (i < trajContactIdx + 3)
            {
                col = TFT_YELLOW;
            }
            else
            {
                col = 0x0600;
            }
            tft.drawLine((int16_t)trajPrevPxX[i], (int16_t)trajPrevPxY[i],
                         (int16_t)trajPrevPxX[i + 1], (int16_t)trajPrevPxY[i + 1], col);
        }
        if (trajContactIdx >= 0 && trajContactIdx < trajLen)
        {
            tft.drawCircle((int16_t)trajPrevPxX[trajContactIdx],
                           (int16_t)trajPrevPxY[trajContactIdx], 5, TFT_WHITE);
        }
        prevTrajContactIdx = trajContactIdx;
        prevTrajLen = trajLen;
    }
    else
    {
        prevTrajLen = 0;
        prevTrajContactIdx = -1;
    }

    // --- Puck (no velocity arrow) ---
    if (puckValid)
    {
        int16_t px = mmToPxX(puckX);
        int16_t py = mmToPxY(puckY);
        tft.fillCircle(px, py, 6, COL_PUCK);
        tft.drawCircle(px, py, 7, 0x0400);
        prevPuckPX = px;
        prevPuckPY = py;
    }

    // --- Mallet ---
    if (malletValid)
    {
        int16_t mx = mmToPxX(malletX);
        int16_t my = mmToPxY(malletY);
        tft.drawCircle(mx, my, 8, COL_MALLET);
        tft.drawCircle(mx, my, 7, COL_MALLET);
        tft.fillCircle(mx, my, 3, COL_MALLET);
        prevMalletPX = mx;
        prevMalletPY = my;
    }
}

void updateGameView()
{
    eraseAllDynamic();
    drawAllDynamic();
}

// ===================== GOAL BANNER ================
//
// While goalBannerEndMs > 0 and millis() < goalBannerEndMs, we draw a
// big banner across the middle of the table announcing who scored, and
// suppress the per-tick puck/mallet redraw so it doesn't get covered.
//
// When the banner expires, we wipe the table area to black, repaint the
// static lines, and reset the prev* dynamic-position trackers so the
// next "state" message redraws everything cleanly.

void drawGoalBanner()
{
    int16_t bw = TABLE_PX_W - 40;
    int16_t bh = 70;
    int16_t bx = TABLE_PX_X + 20;
    int16_t by = TABLE_PX_Y + (TABLE_PX_H - bh) / 2;

    tft.fillRoundRect(bx, by, bw, bh, 12, TFT_BLACK);
    tft.drawRoundRect(bx, by, bw, bh, 12, goalBannerColor);
    tft.drawRoundRect(bx + 1, by + 1, bw - 2, bh - 2, 11, goalBannerColor);

    tft.setTextDatum(MC_DATUM);
    tft.setTextColor(goalBannerColor, TFT_BLACK);
    tft.drawString(goalBannerText, bx + bw / 2, by + bh / 2, 4);
}

void clearGoalBanner()
{
    // Wipe the inside of the table area and repaint the static lines.
    tft.fillRect(TABLE_PX_X, TABLE_PX_Y, TABLE_PX_W, TABLE_PX_H, TFT_BLACK);
    redrawTableLines();

    // Force the next state message to fully repaint puck/mallet/trajectory.
    prevPuckPX = -1;
    prevMalletPX = -1;
    prevTrajLen = 0;
    prevTrajContactIdx = -1;
}

bool goalBannerActive()
{
    return goalBannerEndMs > 0 && (long)(millis() - goalBannerEndMs) < 0;
}

void serviceGoalBanner()
{
    // Called every loop tick. Clears the banner once it expires.
    if (goalBannerEndMs > 0 && (long)(millis() - goalBannerEndMs) >= 0)
    {
        goalBannerEndMs = 0;
        if (currentScreen == SCREEN_GAME)
        {
            clearGoalBanner();
        }
    }
}

void triggerGoalBanner(const char *who)
{
    if (strcmp(who, "robot") == 0)
    {
        snprintf(goalBannerText, sizeof(goalBannerText), "ROBOT SCORES!");
        goalBannerColor = COL_OUR_GOAL;
    }
    else
    {
        snprintf(goalBannerText, sizeof(goalBannerText), "PLAYER SCORES!");
        goalBannerColor = COL_THEIR_GOAL;
    }
    goalBannerEndMs = millis() + GOAL_BANNER_MS;
    if (goalBannerEndMs == 0)
        goalBannerEndMs = 1; // avoid the "disabled" sentinel
    if (currentScreen == SCREEN_GAME)
    {
        drawScoreBar();
        drawGoalBanner();
    }
}

// ===================== COMMUNICATION ==============

// Forward decl
void processIncomingMessage(const char *buf);

void broadcastMessage(const char *json)
{
    // Newline-terminated JSON over USB UART.
    Serial.println(json);
}

void sendStart()
{
    const char *diffStr = "none";
    switch (selectedDifficulty)
    {
    case DIFF_EASY:
        diffStr = "easy";
        break;
    case DIFF_MEDIUM:
        diffStr = "medium";
        break;
    case DIFF_HARD:
        diffStr = "hard";
        break;
    default:
        return;
    }
    char msg[96];
    snprintf(msg, sizeof(msg),
             "{\"type\":\"start\",\"difficulty\":\"%s\"}", diffStr);
    broadcastMessage(msg);
}

// Simple JSON value parser (avoids pulling in ArduinoJson)
// Finds "key":value and returns the value as float, or def if not found
float jsonFloat(const char *buf, const char *key, float def)
{
    char pattern[32];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(buf, pattern);
    if (!p)
        return def;
    p += strlen(pattern);
    return atof(p);
}

// Finds "key":"value" and copies value into dst
void jsonString(const char *buf, const char *key, char *dst, int maxLen)
{
    char pattern[32];
    snprintf(pattern, sizeof(pattern), "\"%s\":\"", key);
    const char *p = strstr(buf, pattern);
    if (!p)
        return;
    p += strlen(pattern);
    const char *end = strchr(p, '"');
    if (!end)
        return;
    int len = min((int)(end - p), maxLen - 1);
    strncpy(dst, p, len);
    dst[len] = '\0';
}

// Parse flat array "key":[v0,v1,v2,...] into x/y pair arrays
// Returns number of points parsed
int jsonFloatArray(const char *buf, const char *key, float *outX, float *outY, int maxPts)
{
    char pattern[32];
    snprintf(pattern, sizeof(pattern), "\"%s\":[", key);
    const char *p = strstr(buf, pattern);
    if (!p)
        return 0;
    p += strlen(pattern);

    int count = 0;
    while (count < maxPts)
    {
        // Read X
        while (*p == ' ')
            p++;
        if (*p == ']' || *p == '\0')
            break;
        outX[count] = atof(p);
        p = strchr(p, ',');
        if (!p)
            break;
        p++;
        // Read Y
        while (*p == ' ')
            p++;
        outY[count] = atof(p);
        count++;
        // Advance past this value
        const char *next = strchr(p, ',');
        if (!next || strchr(p, ']') < next)
            break; // ] before next comma = end
        p = next + 1;
    }
    return count;
}

void processIncomingMessage(const char *buf)
{
    char type[16] = "";
    jsonString(buf, "type", type, sizeof(type));

    if (strcmp(type, "state") == 0)
    {
        puckX = jsonFloat(buf, "px", puckX);
        puckY = jsonFloat(buf, "py", puckY);
        puckVX = jsonFloat(buf, "pvx", puckVX);
        puckVY = jsonFloat(buf, "pvy", puckVY);
        puckValid = (jsonFloat(buf, "pv", 0) > 0.5f);
        malletX = jsonFloat(buf, "mx", malletX);
        malletY = jsonFloat(buf, "my", malletY);
        malletValid = (jsonFloat(buf, "mv", 0) > 0.5f);
        jsonString(buf, "strategy", strategy, sizeof(strategy));

        // Parse trajectory if present
        trajLen = jsonFloatArray(buf, "traj", trajX, trajY, MAX_TRAJ_PTS);
        trajContactIdx = (int)jsonFloat(buf, "tc", -1);

        int newScoreUs = (int)jsonFloat(buf, "su", scoreUs);
        int newScoreThem = (int)jsonFloat(buf, "st", scoreThem);
        bool scoreChanged = (newScoreUs != scoreUs || newScoreThem != scoreThem);
        scoreUs = newScoreUs;
        scoreThem = newScoreThem;

        if (currentScreen == SCREEN_GAME)
        {
            if (scoreChanged)
                drawScoreBar();
            // Don't repaint dynamic content over the banner.
            if (!goalBannerActive())
                updateGameView();
        }
    }
    else if (strcmp(type, "score") == 0)
    {
        scoreUs = (int)jsonFloat(buf, "su", scoreUs);
        scoreThem = (int)jsonFloat(buf, "st", scoreThem);
        if (currentScreen == SCREEN_GAME)
            drawScoreBar();
    }
    else if (strcmp(type, "goal") == 0)
    {
        // Update score and pop the 3-second banner.
        scoreUs = (int)jsonFloat(buf, "su", scoreUs);
        scoreThem = (int)jsonFloat(buf, "st", scoreThem);
        char by[16] = "";
        jsonString(buf, "by", by, sizeof(by));
        triggerGoalBanner(by);
    }
}

// Read newline-delimited JSON from the USB UART and dispatch it through the
// shared message handler. Non-JSON lines (boot banners, stray debug prints,
// etc.) are ignored.
void checkSerialIncoming()
{
    static char buf[1024];
    static int buflen = 0;

    while (Serial.available())
    {
        int c = Serial.read();
        if (c < 0)
            break;
        if (c == '\n' || c == '\r')
        {
            if (buflen > 0)
            {
                buf[buflen] = '\0';
                if (buf[0] == '{')
                    processIncomingMessage(buf);
                buflen = 0;
            }
        }
        else if (buflen < (int)sizeof(buf) - 1)
        {
            buf[buflen++] = (char)c;
        }
        else
        {
            // Overflow: drop the partial line.
            buflen = 0;
        }
    }
}

// ===================== TOUCH ======================

void handleMenuTouch()
{
    uint16_t tx, ty;
    bool pressed = getTouch(tx, ty);

    if (pressed && !lastTouchState)
    {
        Difficulty prev = selectedDifficulty;

        if (touchInside(btnEasy, tx, ty))
        {
            selectedDifficulty = DIFF_EASY;
        }
        else if (touchInside(btnMedium, tx, ty))
        {
            selectedDifficulty = DIFF_MEDIUM;
        }
        else if (touchInside(btnHard, tx, ty))
        {
            selectedDifficulty = DIFF_HARD;
        }
        else if (touchInside(btnStart, tx, ty))
        {
            if (selectedDifficulty != DIFF_NONE)
            {
                tft.fillRoundRect(btnStart.x, btnStart.y, btnStart.w, btnStart.h, 8, TFT_WHITE);
                tft.setTextColor(COL_START);
                tft.setTextDatum(MC_DATUM);
                tft.drawString("SENT!", btnStart.x + btnStart.w / 2,
                               btnStart.y + btnStart.h / 2, 4);
                sendStart();
                delay(400);
                // Switch to game screen
                currentScreen = SCREEN_GAME;
                drawTableStatic();
                return;
            }
        }

        if (prev != selectedDifficulty)
        {
            drawButton(btnEasy, selectedDifficulty == DIFF_EASY);
            drawButton(btnMedium, selectedDifficulty == DIFF_MEDIUM);
            drawButton(btnHard, selectedDifficulty == DIFF_HARD);
        }
    }
    lastTouchState = pressed;
}

void sendCommand(const char *cmd)
{
    // Send a command to the laptop: {"type":"pause"} or {"type":"stop"}
    char msg[64];
    snprintf(msg, sizeof(msg), "{\"type\":\"%s\"}", cmd);
    broadcastMessage(msg);
}

void returnToMenu()
{
    currentScreen = SCREEN_MENU;
    prevPuckPX = -1;
    prevMalletPX = -1;
    prevTrajLen = 0;
    goalBannerEndMs = 0;
    drawMenuUI();
}

void handleGameTouch()
{
    uint16_t tx, ty;
    bool pressed = getTouch(tx, ty);

    if (pressed && !lastTouchState)
    {
        if (touchInside(btnPause, tx, ty))
        {
            // Toggle pause
            sendCommand("pause");
            // Flash feedback
            tft.fillRoundRect(btnPause.x, btnPause.y, btnPause.w, btnPause.h, 8, TFT_WHITE);
            tft.setTextColor(TFT_BLACK);
            tft.setTextDatum(MC_DATUM);
            tft.drawString("PAUSE", btnPause.x + btnPause.w / 2,
                           btnPause.y + btnPause.h / 2, 2);
            delay(200);
            drawButton(btnPause, false);
        }
        else if (touchInside(btnStop, tx, ty))
        {
            // Stop game, return to menu
            sendCommand("stop");
            tft.fillRoundRect(btnStop.x, btnStop.y, btnStop.w, btnStop.h, 8, TFT_WHITE);
            tft.setTextColor(TFT_RED);
            tft.setTextDatum(MC_DATUM);
            tft.drawString("STOPPING", btnStop.x + btnStop.w / 2,
                           btnStop.y + btnStop.h / 2, 2);
            delay(500);
            returnToMenu();
        }
    }
    lastTouchState = pressed;
}

// ===================== TOUCH CALIBRATION ==========

#ifdef CALIBRATE_TOUCH
void runCalibration()
{
    uint16_t calData[5];
    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.setTextDatum(MC_DATUM);
    tft.drawString("Tap the corners", SCREEN_W / 2, SCREEN_H / 2, 4);
    tft.calibrateTouch(calData, TFT_MAGENTA, TFT_BLACK, 15);
    Serial.printf("Touch cal: { %d, %d, %d, %d, %d }\n",
                  calData[0], calData[1], calData[2], calData[3], calData[4]);
    Serial.println("Copy these values into touchCalData[] and disable CALIBRATE_TOUCH.");
    for (int i = 0; i < 5; i++)
        touchCalData[i] = calData[i];
}
#endif

// ===================== MAIN =======================

void setup()
{
    Serial.begin(SERIAL_BAUD);
    Serial.setRxBufferSize(2048); // headroom for 50 Hz state messages
    Serial.println("Air Hockey Display starting...");

    // Turn on backlight
    pinMode(LCD_BL, OUTPUT);
    digitalWrite(LCD_BL, HIGH);

    tft.init();
    tft.setRotation(3); // landscape, USB on right
    tft.setTouch(touchCalData);

#ifdef CALIBRATE_TOUCH
    runCalibration();
#endif

    drawMenuUI();
}

void loop()
{
    if (currentScreen == SCREEN_MENU)
    {
        handleMenuTouch();
    }
    else
    {
        handleGameTouch();
        serviceGoalBanner(); // expires the 3-second goal banner if it's done
    }

    checkSerialIncoming();
}
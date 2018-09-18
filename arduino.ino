//Includes for BLE UART library
#include <Arduino.h>
#include <SPI.h>
#include <LiquidCrystal.h>

LiquidCrystal lcd(22, 23, 27, 26, 25, 24);

#define BUTTON_LEFT					51
#define BUTTON_CENTER				52
#define BUTTON_RIGHT				53
#define LCD_WIDTH					16
#define MAX_TRACK					15

// LCD Screens
#define PAGE_DEFAULT				0
#define PAGE_EDIT					1
#define PAGE_ADD					2
#define PAGE_EDIT_COLOR				3
#define PAGE_PATCH					4

//pedal state variables
int sel_effect = -1;
int effects_len = -1;
char **effect_names = NULL;	 	//array of strings, names of effects
byte **effect_colors = NULL;	//array of 3-tuple bytes for colors
int pots[10];					//potentiometer values

//remap analog inputs for buttons because broken pin
int analog_pins[] = {0, 1, 10, 3, 4, 5, 6, 7, 8, 9};

//page / button variables
byte page = PAGE_DEFAULT;
byte button_releases[3] = {0, 0, 0};

// A small helper
void error(const __FlashStringHelper*err) {
	Serial.println(err);
	while (1);
}

void setup() {
	//init serial
	Serial.begin(115200);
	//init RGB / Button pins
	// put your setup code here, to run once:
	pinMode(2, OUTPUT);
	pinMode(3, OUTPUT);
	pinMode(4, OUTPUT);
	pinMode(BUTTON_LEFT, INPUT_PULLUP);
	pinMode(BUTTON_CENTER, INPUT_PULLUP);
	pinMode(BUTTON_RIGHT, INPUT_PULLUP);
	//setup lcd screen
	lcd.begin(LCD_WIDTH, 2);
	lcd.print("    TRY ME");
	//init color to white
	analogWrite(2, 150);		//B
	analogWrite(3, 0);		//G
	analogWrite(4, 150);		//R
}
//returns first valid char from serial port
int serial_read_char(){
	int val = -1;
	while(val < 0)
		val = Serial.read();
	return val;
}
//read a whole line of chars until newline
void serial_read_line(char* buf, int max_len){
	int val = serial_read_char();;
	int ind = 0;
	while(val != 10 && ind < max_len){		//while not newline and in array
		buf[ind++] = val;
		val = serial_read_char();
	}
	for(int i = ind; i < max_len; i++){
		buf[i] = 0;
	}
}
//reads in a list of effects info
void serial_read_effects(){
	//get length first (from hex value, max 255)
	int mods = serial_read_char();
	// Serial.print("ECHO ");
	// Serial.println(mods);
	if(mods == 0){
		effects_len = mods;
		return;
	}
	//allocate space (this assumes I either reset / clean up after myself)
	effect_names = (char**)malloc(sizeof(char*) * mods);
	effect_colors= (byte**)malloc(sizeof(byte*) * mods);
	//get effects names
	int i = 0;
	while(i < mods){
		effect_names[i] = (char*)malloc(sizeof(char) * (LCD_WIDTH + 1));
		memset(effect_names[i], 0, LCD_WIDTH+1);	//clear the buffer
		serial_read_line(effect_names[i], LCD_WIDTH);
		// Serial.print("ECHO NAME "); Serial.println(effect_names[i]);
		i++;
	}
	//get effects colors
	i = 0;
	while(i < mods){
		effect_colors[i] = (byte*)malloc(sizeof(byte) * 3);
		for(int j = 0; j < 3; j++){
			effect_colors[i][j] = serial_read_char();
		}
		// Serial.print(F("ECHO COLOR "));
		// Serial.print(effect_colors[i][0], HEX);
		// Serial.print(effect_colors[i][1], HEX);
		// Serial.println(effect_colors[i][2], HEX);
		i++;
	}
	effects_len = mods;
}
//set the color of the lcd given a byte array
void color_lcd(byte* colors){
	analogWrite(2, 255 - colors[2]);		//B
	analogWrite(3, 255 - colors[1]);		//G
	analogWrite(4, 255 - colors[0]);		//R
}
//print a char array in hex
void print_array(char *arr, int arr_len){
	for(int i = 0; i < arr_len; i++){
		Serial.print("0x");
		Serial.print(arr[i], HEX);
		Serial.print(' ');
	}
	Serial.println(' ');
}
//send all potentiometer values in 10 * 10 = 100 bits
void sendPotValues(){
	static int lcd_timeout = 10;
	//send the pots we're tracking
	char buf[3];
	Serial.print("UPD ");
	Serial.print(sel_effect);
	Serial.print(" ");
	for(int i = 0; i < 10; i++){
		Serial.print(pots[i]);
		Serial.print(" ");
	}
	Serial.print("\n");
	//get back the response string, print if not empty
	char resp[17];
	serial_read_line(resp, 16);
	if(strlen(resp) > 0 && lcd_timeout < 0){
		lcd.clear();
		lcd.setCursor(0,0);
		//put name in center
		int left_pad = (16 - strlen(resp)) / 2;
		for(int i = 0; i < left_pad; i++){ lcd.print(" "); }
		lcd.print(resp);
		lcd_timeout = 10;
	}
	lcd_timeout--;
}
//returns which button was pressed (as in the MACRO), on release
byte buttonPressed(int button){
	//poll buttons
	int val = digitalRead(button);
	if(val != LOW){
		button_releases[button - BUTTON_LEFT] = 1;
	}else if(button_releases[button - BUTTON_LEFT] == 1){
		button_releases[button - BUTTON_LEFT] = 0;
		return 1;
	}
	return 0;
}

void buttonPrint(int button){
//  if(buttonPressed(button)){
//    Serial.print("BTN ");
//  }else{
//    Serial.print("0 ");
//  }
Serial.print(digitalRead(button));
}

void loop() {
	// put your main code here, to run repeatedly:
	//delay(20);
	//read potentiometer values
	int pot_val = 0;
	int diff = 0;
	for(int i = 0; i < 10; i++){
		Serial.print(analogRead(i));
		Serial.print(' ');
	}
  buttonPrint(BUTTON_LEFT);
  buttonPrint(BUTTON_CENTER);
  buttonPrint(BUTTON_RIGHT);
	Serial.println(' ');
//	//initialize if needed
//	if(effects_len < 0){
//		//send command to get current modules
//		//Serial.print("CUR\n");
//		//serial_read_effects();
//	//edit screen, edit a module
//	}else if(page == PAGE_EDIT){
//		//send the potentiometer values
//		sendPotValues();
//		//show the screen
//		lcd.setCursor(0,1);
//		lcd.print("(#)  (back)  (^)");
//		//poll buttons
//		if(buttonPressed(BUTTON_CENTER)){
//			page = PAGE_DEFAULT;
//			//reset effects
//			sel_effect = -1;
//			effects_len = -1;
//		}
//	//add a new effect to current effects
//	}else if(page == PAGE_ADD){
//		int new_sel = analogRead(analog_pins[0]) / (900 / (effects_len));
//		//switch if effect changes
//		if(new_sel != sel_effect && new_sel < effects_len){
//			sel_effect = new_sel;
//			//color the lcd
//			color_lcd(effect_colors[sel_effect]);
//			//display name
//			lcd.clear();
//			lcd.setCursor(0,0);
//			//put name in center
//			int left_pad = (16 - strlen(effect_names[sel_effect])) / 2;
//			for(int i = 0; i < left_pad; i++){ lcd.print(" "); }
//			lcd.print(effect_names[sel_effect]);
//			//print button indicator
//			lcd.setCursor(0,1);
//			lcd.print("(add)     (back)");
//		}
//		//add the effect
//		if(buttonPressed(BUTTON_LEFT)){
//			Serial.print("ADD ");
//			Serial.println(effect_names[sel_effect]);
//			page = PAGE_DEFAULT;
//			//reset effects
//			sel_effect = -1;
//			effects_len = -1;
//		}
//		//cancel
//		if(buttonPressed(BUTTON_RIGHT)){
//			page = PAGE_DEFAULT;
//			//reset effects
//			sel_effect = -1;
//			effects_len = -1;
//		}
//	//default screen, scroll through current effects
//	}else{
//		if(effects_len > 0){	//if we have effects
//			int new_sel = analogRead(analog_pins[0]) / (900 / (effects_len));
//			//switch if effect changes
//			if(new_sel != sel_effect && new_sel < effects_len){
//				sel_effect = new_sel;
//				//color the lcd
//				color_lcd(effect_colors[sel_effect]);
//				//display name
//				lcd.clear();
//				lcd.setCursor(0,0);
//				//put name in center
//				int left_pad = (16 - strlen(effect_names[sel_effect])) / 2;
//				for(int i = 0; i < left_pad; i++){ lcd.print(" "); }
//				lcd.print(effect_names[sel_effect]);
//				//print button indicator
//				lcd.setCursor(0,1);
//				lcd.print("(+)  (edit)  (-)");
//			}
//			if(buttonPressed(BUTTON_CENTER)){
//				page = PAGE_EDIT;
//			}
//		}else{		//if we don't have effects
//			byte white[3] = {255, 255, 255};
//			color_lcd(white);
//			lcd.clear();
//			lcd.setCursor(0,0);
//			//put name in center
//			char* empty_str = "No effects";
//			int left_pad = (16 - strlen(empty_str)) / 2;
//			for(int i = 0; i < left_pad; i++){ lcd.print(" "); }
//			lcd.print(empty_str);
//			//print button indicator
//			lcd.setCursor(0,1);
//			lcd.print("(+)             ");
//			delay(100);		//to not overload the screen refreshing
//		}
//		if(buttonPressed(BUTTON_LEFT)){
//			page = PAGE_ADD;
//			//get all possible effects to add
//			Serial.print("LST\n");
//			serial_read_effects();
//			//reset selected
//			sel_effect = -1;
//		}
//	}
}

'use strict';

var jumpToSlide = "";

function keyHandle(event) {
	var isSpecialKey = event.shiftKey || event.ctrlKey || event.altKey || event.metaKey;
	var isNumberKey = event.which >= 48 && event.which <= 57;
	var isDashKey = event.which === 45;

	if (isNumberKey || isDashKey && !isSpecialKey) {
		jumpToSlide += String.fromCharCode(event.charCode);
	} else {
		var isEnterKey = event.which === 13;
		var isJumpToSlideEmpty = jumpToSlide === "";

		if (isEnterKey && !isJumpToSlideEmpty) {
			// horizontal and vertical slides are separated by a dash
			jumpToSlide = jumpToSlide.split("-");

			Reveal.slide(jumpToSlide[0], jumpToSlide[1]);

			// Reset jumpToSlide variable
			jumpToSlide = "";
		}
	}
}

document.onkeypress = keyHandle;


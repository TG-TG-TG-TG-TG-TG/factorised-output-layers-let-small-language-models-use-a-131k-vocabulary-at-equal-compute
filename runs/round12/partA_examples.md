# Round 12 Part A: ten LAMBADA passages with large F8k vs H3* last-word differences

Drawn with np.random.default_rng(12) from the 100 passages with the largest |F8k - H3*| last-word bits/byte. Per-token NLL in bits; `|` separates tokens.

## passage 1372 (appears earlier: yes; F8k 1.992 vs H3* 5.650 bits/byte)

> …this job for you," he said gently. "You will understand one day, when you are a parent."  Without saying anything more, her father started to walk away. He turned to look back at her solemnly before climbing the stairs. "We'll be moving during your winter break. I know how hard this will be for you, ** Ivy**

- H3s384_575M: `' Ivy'` 22.60
- F8k384_631M: `' I'` 4.70 | `'v'` 3.20 | `'y'` 0.07

## passage 1762 (appears earlier: yes; F8k 0.822 vs H3* 3.770 bits/byte)

> …want your stupid revenge!” Ember’s eyes widened, her temples pulling taut against her skull. She crossed the room in two steps, the back of her hand whipping across Ragnar’s face. Ragnar fell against the bedside table and pressed a hand to his cheek. “Take him to the Air District headquarters,” said ** Ember**

- H3s384_575M: `' Ember'` 22.62
- F8k384_631M: `' E'` 4.84 | `'mber'` 0.09

## passage 1987 (appears earlier: yes; F8k 2.141 vs H3* 5.083 bits/byte)

> …Vo grabbed a cold bottle of water from the refrigerator. He looked at the dish rack full of wet plates, bowls and utensils. “Was Kim here?” “Yes, she was. We were both sorry that you had to miss dinner, again.” “I told you my schedule is erratic,” said ** Vo**

- H3s384_575M: `' Vo'` 15.25
- F8k384_631M: `' V'` 5.97 | `'o'` 0.45

## passage 2219 (appears earlier: yes; F8k 1.293 vs H3* 3.938 bits/byte)

> …Fareed hadn't seen anything like that in the man's eyes since his late wife.  The gregarious Rose eyed him back with open appreciation and murmured to Gwen for all to hear, "So incredible things do come to those who wait, eh, sweetie?" She didn't wait for Gwen's reaction and turned to ** Fareed**

- H3s384_575M: `' Fare'` 17.96 | `'ed'` 9.61
- F8k384_631M: `' F'` 7.92 | `'are'` 0.55 | `'ed'` 0.58

## passage 2329 (appears earlier: yes; F8k 3.698 vs H3* 6.107 bits/byte)

> …I turned to Kel with a questioning glance. "Do you want to have the discussion I deferred now or later? Are you on a schedule?"  "Don't worry about it. I've withstood many punishments over the years."  Guilt flared in a hard, awful twinge. "I don't want to be the reason you get hurt, ** Kel**

- H3s384_575M: `' Kel'` 24.43
- F8k384_631M: `' K'` 14.45 | `'el'` 0.34

## passage 2514 (appears earlier: yes; F8k 0.928 vs H3* 3.119 bits/byte)

> …"She hasn't told us. And she tells us everything. Or she did." Della shot Kylie a frown.  "She doesn't tell us everything," Miranda piped up. "Just like some people with hickeys don't tell us everything."  Della scowled at Miranda and then looked at ** Kylie**

- H3s384_575M: `' Kyl'` 18.71 | `'ie'` 0.00
- F8k384_631M: `' K'` 5.44 | `'yl'` 0.10 | `'ie'` 0.02

## passage 2961 (appears earlier: yes; F8k 1.996 vs H3* 4.905 bits/byte)

> …Mareth was her coach, and he wasn't afraid to chew her out.  "Oh, Mareth, how long are you to stay angry with me?" said Luxa. "It has been at least a day already. No one will blame you for my disobedience."  "That is much beside the point, ** Luxa**

- H3s384_575M: `' Lux'` 24.50 | `'a'` 0.03
- F8k384_631M: `' L'` 9.76 | `'ux'` 0.15 | `'a'` 0.07

## passage 4483 (appears earlier: yes; F8k 2.360 vs H3* 4.795 bits/byte)

> …p at the ceiling as tears stung the back of my eyes for reasons I didn’t even understand myself. “Em,” Gabe said taking notice of my tearing eyes. Gently he dabbed them away with his thumb and peered down at me with concern.  What’s wrong?”  He whispered. I glanced at the flower quickly then back at ** Gabe**

- H3s384_575M: `' Gabe'` 23.97
- F8k384_631M: `' G'` 11.52 | `'ab'` 0.27 | `'e'` 0.01

## passage 4692 (appears earlier: yes; F8k 2.516 vs H3* 4.710 bits/byte)

> …In fact, they went to Atlanta to see the different types of carriages so they could settle on the type they will be using. I’m sorry.” Wade nodded, understanding, and replied, “Thank you for your time.” He rose to leave. Mr. Johnson stopped him, “Please sit down, ** Wade**

- H3s384_575M: `' Wade'` 23.55
- F8k384_631M: `' W'` 11.90 | `'ade'` 0.68

## passage 4785 (appears earlier: yes; F8k 1.959 vs H3* 4.784 bits/byte)

> …Greg scans the area, taking in the size of the lake. "Now that’s a big lake," he says, with the excitement of a true fisherman. He strolls over to the dock and checks out the area as the others get their hunting gear ready. "Here you go," Bob says, tossing the keys to ** Greg**

- H3s384_575M: `' Greg'` 23.92
- F8k384_631M: `' G'` 9.48 | `'reg'` 0.32


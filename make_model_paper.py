import json

model_paper = {

	'1' : {
		'q' : 'Jatis are a reflection of the complexities that existed within the social categories. Justify.',
		'a' : 'Jatis like varna based on birth but unlike varna had no limitation on its number. New groups like the nishada or the goldsmith who could not be adjusted in the varna fold were assigned the status of jati. Common profession shared by jatis were called shrenis. Stone inscription of Mandasor in Madhya Pradesh, tells us that though generally occupation was common but a few people were of other professions, these people would also commonly invest to construct temples.',
		'm' : 5
		},

	'2' : {
		'q' : 'The simple Buddha had a simple following. Justify.',
		'a' : 'a. Monks living in the sangha were very simple, received food in a bowl and took alms for which they were called Bhikkus. Importance was placed on conduct rather than birth, they considered emotions of compassion and fellow feeling as their real wealth. A rug made by the Bhikku was to be used for six years and in case he has a new one before the aforesaid time it would be forfeited. If a Bhikku is presented with good meals at his residence, he is to accept only two or three bowlfuls and whatever they used in the sangha had to be kept back at the place from where they were taken. The very fact that the Bhikkus followed all this showed that they ready for a simple life.',
		'm' : 5
		},

	'3' : {
		'q' : 'Who were Lingayats? Explain their contribution in the social and religious fields with special reference to the caste system.',
		'a' : 'i.Lingayat a popular movement which emerged during the twelth century. Its founder was Basava and they established their faith after disputes with the Jainas. Were worshippers of Lord Shiva who strongly opposed the caste system, rejected fasts, feats, pilgrimages and sacrifice. In the social sphere they opposed child marriage and allowed remarriage of the widows.',
		'm' : 5
	}
}

with open('history_model_paper.json', 'w') as j:
	json.dump(model_paper, j, indent=4)


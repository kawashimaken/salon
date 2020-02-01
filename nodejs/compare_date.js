/*
日付比較（有効期限が過ぎているかどうか）
*/
var moment = require('moment-timezone');

var datetime = '2018-06-04 12:20:43 Etc/GMT'.split(' ');
var expires_date = moment
    .tz(`${datetime[0]} ${datetime[1]}`, datetime[2])
    .clone()
    .tz('Asia/Tokyo')
    .format('YYYY-MM-DD HH:mm:ss');
var today = moment(new Date())
    .clone()
    .tz('Asia/Tokyo')
    .format('YYYY-MM-DD HH:mm:ss');
//

var result_string = '';

if (expires_date >= today) {
    result_string = '有効期限が大きい';
} else {
    result_string = '今日が大きい';
}
console.log('expires_date ->', expires_date);
console.log('today -->', today);
console.log(result_string);

import React from 'react';
import './App.css';
// secret_exampl.jsのように、twilioのAPIなどを設定しておいてください
// 用意したファイルを「secret.js」にしてください
import {ACCOUNT_SID,API_KEY_SID,API_KEY_SECRET} from './secret'

const {
  connect
} = require('twilio-video');

var AccessToken = require('twilio').jwt.AccessToken;
var VideoGrant = AccessToken.VideoGrant;

// Create an Access Token
var accessToken = new AccessToken(
  ACCOUNT_SID,
  API_KEY_SID,
  API_KEY_SECRET
);

// Set the Identity of this token
accessToken.identity = 'example-user';

// Grant access to Video
var grant = new VideoGrant();
grant.room = 'DemoRoom';
accessToken.addGrant(grant);

// Serialize the token as a JWT
var jwt = accessToken.toJwt();
console.log(jwt);
//

connect(jwt, {
  name: 'DemoRoom'
}).then(room => {
  console.log(`Successfully joined a Room: ${room}`);
  //
  room.on('participantConnected', participant => {
    console.log(`A remote Participant connected: ${participant}`);
  });
}, error => {
  console.error(`Unable to connect to Room: ${error.message}`);
});

//
// Set the Identity of this token
accessToken.identity = 'another-user';

// Grant access to Video
var grant = new VideoGrant();
grant.room = 'DemoRoom';
accessToken.addGrant(grant);

// Serialize the token as a JWT
var another_jwt = accessToken.toJwt();
console.log(another_jwt);

//
connect(another_jwt, {
  name: 'DemoRoom'
}).then(room => {
  console.log(`Another User Successfully joined a Room: ${room}`);
  //
  room.on('participantConnected', participant => {
    console.log(`Another User said: A remote Participant connected: ${participant}`);
  });
}, error => {
  console.error(`Another User is Unable to connect to Room: ${error.message}`);
});


function App() {
  return (
    <div> Hello </div>);
}

export default App;
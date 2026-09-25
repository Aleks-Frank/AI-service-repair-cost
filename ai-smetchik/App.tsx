import React from 'react';
import { NavigationContainer, DefaultTheme } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import { colors } from './src/theme';
import { StoreProvider } from './src/store';
import { Dashboard } from './src/screens/Dashboard';
import { NewRequest } from './src/screens/NewRequest';
import { Processing } from './src/screens/Processing';
import { Estimate } from './src/screens/Estimate';
import type { RootStackParamList } from './src/navigation/types';

const Stack = createNativeStackNavigator<RootStackParamList>();

const navTheme = {
  ...DefaultTheme,
  colors: { ...DefaultTheme.colors, background: colors.bg, primary: colors.primary, card: colors.bg, text: colors.text, border: colors.border },
};

export default function App() {
  return (
    <StoreProvider>
      <NavigationContainer theme={navTheme}>
        <StatusBar style="dark" />
        <Stack.Navigator screenOptions={{ headerShown: false, animation: 'slide_from_right' }}>
          <Stack.Screen name="Dashboard" component={Dashboard} />
          <Stack.Screen name="NewRequest" component={NewRequest} />
          <Stack.Screen name="Processing" component={Processing} options={{ animation: 'fade' }} />
          <Stack.Screen name="Estimate" component={Estimate} />
        </Stack.Navigator>
      </NavigationContainer>
    </StoreProvider>
  );
}
